"""Tests for edgevision.evaluation.benchmark_plots and scripts/plot_benchmarks.py.

The result files here are built from hand-made run data; they exercise the
validation and plotting code and are not measurements. matplotlib runs with
the non-interactive Agg backend and every figure is closed.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from edgevision.benchmarks.report import build_report  # noqa: E402
from edgevision.benchmarks.runner import BenchmarkPlan, RunResult  # noqa: E402
from edgevision.evaluation import benchmark_plots as bp  # noqa: E402
from edgevision.optional import MissingDependencyError  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def run(n: int, detect: list[float], stages: bool = True, fps: float | None = None) -> RunResult:
    samples = {"detect": detect}
    if stages:
        samples |= {"preprocess_ms": [5.0] * len(detect), "inference_ms": [d - 7.0 for d in detect],
                    "postprocess_ms": [1.5] * len(detect)}
    wall = len(detect) / fps if fps else sum(detect) / 1000
    return RunResult(run=n, init_s=0.2, first_inference_ms=300.0, warmup_iterations=2, warmup_s=0.3,
                     measured_iterations=len(detect), measured_wall_s=wall, samples_ms=samples, detections_total=40,
                     device="cpu", execution_providers=None)


def result_doc(backend: str = "pytorch", runs=None, **changes) -> dict:
    runs = runs or [run(1, [100.0, 102.0, 98.0, 101.0]), run(2, [99.0, 100.0, 103.0, 97.0])]
    doc = build_report(
        plan=BenchmarkPlan(warmup=2, iterations=4, runs=len(runs)).to_dict(),
        backend={"name": backend, "runtime": backend, "model": {"path": f"models/m.{backend}"}, "device": "cpu",
                 "execution_providers": ["CPUExecutionProvider"] if backend == "onnx" else None,
                 "letterbox": "square", "model_input_shape": [1, 3, 640, 640], "precision": "fp32",
                 "settings": {"input_size": 640, "conf_threshold": 0.25, "iou_threshold": 0.7, "classes": None}},
        input_info={"kind": "video", "file": {"path": "data/input/sample.mp4", "sha256": "ab"}, "width": 1024,
                    "height": 576},
        runs=runs, library_import={}, environment={"cpu": {"model": "Test CPU"}, "os": {"platform": "TestOS"}},
        git=None, warnings=[],
    )
    for key, value in changes.items():
        target = doc
        *parents, last = key.split(".")
        for p in parents:
            target = target[p]
        target[last] = value
    return doc


def save(tmp_path: Path, name: str, doc: dict) -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


@pytest.fixture
def plt():
    pytest.importorskip("matplotlib", reason='matplotlib not installed (pip install -e ".[plots]")')
    import matplotlib.pyplot

    yield matplotlib.pyplot
    assert matplotlib.pyplot.get_fignums() == []  # every figure was closed


# --- validation ---------------------------------------------------------------------------------


def test_valid_document_has_no_problems() -> None:
    assert bp.validate_benchmark(result_doc()) == []


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["summary"]["latency_ms_pooled"].pop("detect"), "missing field summary.latency_ms_pooled.detect"),
        (lambda d: d.pop("environment"), "missing field environment.cpu"),
        (lambda d: d["summary"]["latency_ms_pooled"]["detect"].update(p95=None), "detect.p95 is missing"),
        (lambda d: d["runs"][0]["samples_ms"].update(detect=[]), "runs[0].samples_ms.detect"),
        (lambda d: d["runs"][1].pop("throughput_fps"), "runs[1].throughput_fps"),
        (lambda d: d["summary"].update(throughput_fps_per_run=[1.0]), "one finite number per run"),
        (lambda d: d.update(runs=[]), "runs must be a non-empty list"),
    ],
)
def test_missing_or_broken_fields(mutate, message: str) -> None:
    doc = result_doc()
    mutate(doc)
    assert any(message in p for p in bp.validate_benchmark(doc)), bp.validate_benchmark(doc)


def test_load_errors(tmp_path: Path) -> None:
    with pytest.raises(bp.BenchmarkFileError, match="not found"):
        bp.load_benchmark_files([tmp_path / "nope.json"])
    bad = tmp_path / "bad.json"
    bad.write_text("{oops", encoding="utf-8")
    with pytest.raises(bp.BenchmarkFileError, match="bad.json"):
        bp.load_benchmark_files([bad])
    other = save(tmp_path, "other", {"schema": "edgevision.video_detections", "schema_version": 1})
    with pytest.raises(bp.BenchmarkFileError, match="not an EdgeVision benchmark"):
        bp.load_benchmark_files([other])
    broken = result_doc()
    del broken["summary"]["init_s_per_run"]
    with pytest.raises(bp.BenchmarkFileError, match="missing field summary.init_s_per_run"):
        bp.load_benchmark_files([save(tmp_path, "broken", broken)])
    with pytest.raises(bp.BenchmarkFileError, match="at most 4"):
        bp.load_benchmark_files([save(tmp_path, f"r{i}", result_doc()) for i in range(5)])


def test_same_file_names_get_their_folder(tmp_path: Path) -> None:
    (tmp_path / "laptop").mkdir()
    (tmp_path / "desktop").mkdir()
    results = bp.load_benchmark_files([save(tmp_path / "laptop", "onnx", result_doc("onnx")),
                                       save(tmp_path / "desktop", "onnx", result_doc("onnx"))])
    assert [r.name for r in results] == ["laptop/onnx", "desktop/onnx"]


# --- values ---------------------------------------------------------------------------------------


def test_stage_means_and_other() -> None:
    means = bp.stage_means(result_doc())
    assert means["preprocess_ms"] == 5.0 and means["postprocess_ms"] == 1.5
    assert means["inference_ms"] == pytest.approx(93.0)
    assert means["other"] == pytest.approx(0.5)  # detect mean 100 - (5 + 93 + 1.5)
    no_stages = result_doc(runs=[run(1, [10.0, 11.0], stages=False)])
    assert bp.stage_means(no_stages) is None  # not reported, not zero


def test_context_flags_differences_time_gaps_and_interruptions() -> None:
    a = result_doc("pytorch", created_utc="2026-10-18T10:00:00Z")
    b = result_doc("onnx", **{"backend.model_input_shape": [1, 3, 384, 640], "created_utc": "2026-10-18T13:30:00Z"},
                   runs=[run(1, [95.0, 96.0, 94.0, 9_000_000.0]), run(2, [95.0] * 4)])
    results = [bp.BenchmarkResult("a", Path("a.json"), a), bp.BenchmarkResult("b", Path("b.json"), b)]
    _, notes = bp.context_lines(results)
    text = " ".join(notes)
    assert "NOT DIRECTLY COMPARABLE" in text and "model input tensor differs" in text
    assert "recorded 3.5 h apart" in text
    assert "b: run(s) 1 contain iterations flagged as interrupted" in text
    assert "not a general result" in text and "nothing here measures detection accuracy" in text


# --- figures and summary --------------------------------------------------------------------------


def test_generate_report(tmp_path: Path, plt) -> None:
    results = bp.load_benchmark_files([save(tmp_path, "pytorch_cpu", result_doc("pytorch")),
                                       save(tmp_path, "onnx_cpu", result_doc("onnx"))])
    out = tmp_path / "plots"
    written = bp.generate_report(results, out)
    assert [p.name for p in written] == list(bp.OUTPUT_FILES)
    for path in written[:4]:
        assert path.read_bytes()[:8] == PNG_SIGNATURE and path.stat().st_size > 10_000
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "| | pytorch_cpu | onnx_cpu |" in summary
    assert "| detect median (ms) | 100.00 | 100.00 |" in summary  # median of the 8 pooled samples
    assert "Test CPU" in summary and "not an accuracy measure" in summary
    assert not list(out.glob("*.partial*"))
    with pytest.raises(FileExistsError, match="--overwrite"):
        bp.generate_report(results, out)
    bp.generate_report(results, out, overwrite=True)


def test_missing_stages_and_interrupted_runs_are_shown_not_hidden(tmp_path: Path, plt) -> None:
    doc = result_doc(runs=[run(1, [10.0, 11.0, 12.0, 5_000.0], stages=False)])
    results = bp.load_benchmark_files([save(tmp_path, "odd", doc)])
    bp.generate_report(results, tmp_path / "plots")
    summary = (tmp_path / "plots" / "summary.md").read_text(encoding="utf-8")
    assert "| preprocess median (ms) | not reported |" in summary
    assert "| Runs flagged as interrupted | 1 |" in summary
    assert "| detect mean (ms) | 1258.25 |" in summary  # (10 + 11 + 12 + 5000) / 4: the outlier is not dropped
    assert json.loads((tmp_path / "odd.json").read_text())["runs"][0]["samples_ms"]["detect"][-1] == 5000.0


def test_missing_matplotlib(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    results = bp.load_benchmark_files([save(tmp_path, "r", result_doc())])
    with pytest.raises(MissingDependencyError, match=r"\.\[plots\]"):
        bp.generate_report(results, tmp_path / "plots")


# --- command line -------------------------------------------------------------------------------


def load_script() -> object:
    spec = importlib.util.spec_from_file_location("plot_benchmarks_script", REPO / "scripts" / "plot_benchmarks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli(tmp_path: Path, plt, capsys) -> None:
    a = save(tmp_path, "pytorch_cpu", result_doc("pytorch"))
    b = save(tmp_path, "onnx_cpu", result_doc("onnx"))
    script = load_script()
    assert script.main([str(a), str(b), "--output", str(tmp_path / "out")]) == 0
    assert "latency.png" in capsys.readouterr().out
    assert script.main([str(a), str(b), "--output", str(tmp_path / "out")]) == 1  # refuses to overwrite
    assert "already contains" in capsys.readouterr().err
    assert script.main([str(tmp_path / "missing.json"), "--output", str(tmp_path / "x")]) == 1


def test_cli_help(tmp_path: Path) -> None:
    result = subprocess.run([sys.executable, str(REPO / "scripts" / "plot_benchmarks.py"), "--help"],
                            capture_output=True, text=True, check=False, cwd=tmp_path)
    assert result.returncode == 0 and "--output" in result.stdout and list(tmp_path.iterdir()) == []
