"""Tests for scripts/benchmark.py and scripts/compare_benchmarks.py.

Argument errors and --help run as subprocesses. The other tests call main()
in-process with the fake PyTorch/Ultralytics (``fake_yolo``) and fake ONNX
Runtime (``fake_ort``) from conftest.py on a tiny synthetic video: they check
the command's logic and the result file, and their timings are meaningless.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "default.yaml"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"{name}_bench_script", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(name: str, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(REPO / "scripts" / f"{name}.py"), *args], capture_output=True,
                          text=True, check=False, cwd=cwd)


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A working directory with a model file, a short video, an image and a config."""
    cv2 = pytest.importorskip("cv2", reason='OpenCV not installed (pip install -e ".[vision]")')
    np = pytest.importorskip("numpy")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "yolov8n.onnx").write_bytes(b"fake onnx")
    (tmp_path / "models" / "yolov8n.pt").write_bytes(b"fake pt")
    (tmp_path / "data").mkdir()
    writer = cv2.VideoWriter(str(tmp_path / "data" / "clip.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 120))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot encode MJPG video")
    for _ in range(8):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()
    assert cv2.imwrite(str(tmp_path / "data" / "photo.png"), np.zeros((120, 160, 3), dtype=np.uint8))
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data["video"]["input"] = "data/clip.avi"
    data["benchmark"].update(warmup=1, iterations=3, runs=2, preload_frames=4)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


ARGS = ["--config", "config.yaml", "--quiet"]


# --- command line ------------------------------------------------------------------------------


def test_help_does_not_benchmark(tmp_path: Path) -> None:
    result = run("benchmark", "--help", cwd=tmp_path)
    assert result.returncode == 0
    for option in ("--backend", "--model", "--device", "--scope", "--warmup", "--iterations", "--runs",
                   "--preload-frames", "--tracking", "--pytorch-letterbox", "--conf", "--imgsz", "--output",
                   "--overwrite"):
        assert option in result.stdout
    assert list(tmp_path.iterdir()) == []
    assert run("compare_benchmarks", "--help").returncode == 0


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--iterations", "0"], "must be a positive integer"),
        (["--warmup", "-1"], "must be an integer >= 0"),
        (["--runs", "x"], "must be a positive integer"),
        (["--scope", "model_only"], "invalid choice"),
        (["--backend", "onnx", "--pytorch-letterbox", "minimal"], "applies only to --backend pytorch"),
        (["--backend", "onnx", "--model", "models/yolov8n.pt"], "needs an .onnx file"),
        (["--output", "result.txt"], "--output must end in .json"),
        (["photo.jpg", "--scope", "pipeline"], "needs a video input"),
        (["--tracking"], "only be benchmarked in the pipeline scope"),
        (["--imgsz", "641"], "multiple of 32"),
    ],
)
def test_invalid_arguments_exit_with_status_2(args: list[str], message: str) -> None:
    result = run("benchmark", *args, "--config", str(CONFIG))
    assert result.returncode == 2 and message in result.stderr


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["missing.mp4"], "Input file not found"),
        (["data/clip.avi", "--model", "models/other.pt"], "Benchmarks never download models"),
        (["data/clip.avi", "--backend", "onnx", "--model", "models/none.onnx"], "export_onnx.py"),
    ],
)
def test_missing_files_exit_with_status_1(project: Path, capsys, args: list[str], message: str) -> None:
    assert load("benchmark").main([*args, *ARGS]) == 1
    assert message in capsys.readouterr().err
    assert not (project / "results").exists()


def test_existing_output_is_not_replaced(project: Path, fake_ort, capsys) -> None:
    out = project / "existing.json"
    out.write_text("keep me", encoding="utf-8")
    assert load("benchmark").main(["--backend", "onnx", "--output", "existing.json", *ARGS]) == 1
    assert "already exists" in capsys.readouterr().err
    assert out.read_text(encoding="utf-8") == "keep me"
    assert fake_ort.sessions == []  # nothing was run


# --- mocked runs -------------------------------------------------------------------------------


def test_onnx_detector_scope_writes_a_complete_result(project: Path, fake_ort, capsys) -> None:
    assert load("benchmark").main(["--backend", "onnx", "--device", "cpu", *ARGS]) == 0
    files = list((project / "results" / "benchmarks").glob("onnx_cpu_detector_*.json"))
    assert len(files) == 1  # default output path
    doc = json.loads(files[0].read_text(encoding="utf-8"))

    assert doc["schema"] == "edgevision.benchmark" and doc["schema_version"] == 1
    assert doc["benchmark"] == {"scope": "detector", "warmup_iterations": 1, "measured_iterations": 3, "runs": 2,
                                "preload_frames": 4, "tracking": False, "batch_size": 1}
    b = doc["backend"]
    assert (b["name"], b["device"], b["execution_providers"]) == ("onnx", "cpu", ["CPUExecutionProvider"])
    assert b["model"]["path"] == "models/yolov8n.onnx" and len(b["model"]["sha256"]) == 64
    assert b["model_input_shape"] == [1, 3, 640, 640] and b["letterbox"] == "square"
    assert b["settings"] == {"input_size": 640, "conf_threshold": 0.25, "iou_threshold": 0.7, "classes": None}
    assert doc["input"]["kind"] == "video" and doc["input"]["file"]["path"] == "data/clip.avi"
    assert (doc["input"]["width"], doc["input"]["height"], doc["input"]["frames_in_memory"]) == (160, 120, 4)
    assert len(fake_ort.sessions) == 2  # a new session for each run, reused within the run
    assert len(fake_ort.feeds) == 2 * (1 + 3)  # warm-up + measured calls
    for r in doc["runs"]:
        assert set(r["samples_ms"]) == {"detect", "preprocess_ms", "inference_ms", "postprocess_ms"}
        assert len(r["samples_ms"]["detect"]) == 3 and r["latency_ms"]["detect"]["count"] == 3
        assert r["init_s"] >= 0 and r["throughput_fps"] > 0
    assert doc["summary"]["latency_ms_pooled"]["detect"]["count"] == 6
    assert "executable" not in json.dumps(doc["environment"])
    assert any("p90/p95 are not reliable" in w for w in doc["warnings"])
    out = capsys.readouterr().out
    assert "Benchmark summary" in out and "Saved:" in out


def test_pytorch_uses_square_padding_by_default(project: Path, fake_yolo) -> None:
    script = load("benchmark")
    assert script.main(["--output", "pt.json", *ARGS]) == 0
    assert all(call.get("rect") is False for call in fake_yolo.predict_calls)
    doc = json.loads((project / "pt.json").read_text(encoding="utf-8"))
    assert doc["backend"]["model_input_shape"] == [1, 3, 640, 640] and doc["backend"]["execution_providers"] is None
    # The fake model reports no stage times: they are marked unavailable, not invented.
    assert set(doc["runs"][0]["samples_ms"]) == {"detect"}
    assert any("stage breakdown unavailable" in n for n in doc["runs"][0]["notes"])

    assert script.main(["--pytorch-letterbox", "minimal", "--output", "pt_min.json", *ARGS]) == 0
    doc = json.loads((project / "pt_min.json").read_text(encoding="utf-8"))
    assert doc["backend"]["model_input_shape"] == [1, 3, 480, 640]
    assert any("minimal padding" in w for w in doc["warnings"])


def test_image_input(project: Path, fake_ort) -> None:
    assert load("benchmark").main(["data/photo.png", "--backend", "onnx", "--output", "img.json", *ARGS]) == 0
    doc = json.loads((project / "img.json").read_text(encoding="utf-8"))
    assert doc["input"]["kind"] == "image" and doc["input"]["frames_in_memory"] == 1


def test_pipeline_scope_with_tracking(project: Path, fake_ort, fake_tracker_class, monkeypatch) -> None:
    import edgevision.tracking

    monkeypatch.setattr(edgevision.tracking, "ByteTrackTracker", lambda: fake_tracker_class())
    args = ["--backend", "onnx", "--scope", "pipeline", "--tracking", "--iterations", "20", "--output", "p.json"]
    assert load("benchmark").main([*args, *ARGS]) == 0
    doc = json.loads((project / "p.json").read_text(encoding="utf-8"))
    run1 = doc["runs"][0]
    assert list(run1["samples_ms"])[:4] == ["decode", "detect", "track", "frame"]
    assert run1["measured_iterations"] == 7  # 8 frames - 1 warm-up
    assert any("ended after 7 of 20" in n for n in run1["notes"])
    assert run1["tracker_init_s"] is not None


def test_run_failure_writes_nothing(project: Path, fake_ort, capsys) -> None:
    fake_ort.run_error = RuntimeError("kernel exploded")
    assert load("benchmark").main(["--backend", "onnx", "--output", "x.json", *ARGS]) == 1
    assert "kernel exploded" in capsys.readouterr().err
    assert not (project / "x.json").exists()


def test_interrupt_writes_nothing(project: Path, fake_ort, capsys) -> None:
    fake_ort.run_error = KeyboardInterrupt()
    assert load("benchmark").main(["--backend", "onnx", "--output", "x.json", *ARGS]) == 130
    assert "no results were written" in capsys.readouterr().err
    assert not (project / "x.json").exists()


def test_compare_two_results(project: Path, fake_ort, capsys) -> None:
    script = load("benchmark")
    assert script.main(["--backend", "onnx", "--output", "a.json", *ARGS]) == 0
    assert script.main(["--backend", "onnx", "--iterations", "4", "--output", "b.json", *ARGS]) == 0
    capsys.readouterr()
    compare = load("compare_benchmarks")
    assert compare.main(["a.json", "b.json"]) == 0
    out = capsys.readouterr().out
    assert "a" in out and "b" in out and "measured iterations differs: 3 vs 4" in out


def test_compare_errors(project: Path, capsys) -> None:
    compare = load("compare_benchmarks")
    (project / "bad.json").write_text("{not json", encoding="utf-8")
    assert compare.main(["bad.json", "bad.json"]) == 1
    assert compare.main(["missing.json", "bad.json"]) == 1
    assert "file not found" in capsys.readouterr().err
    with pytest.raises(SystemExit) as info:
        compare.main(["bad.json"])
    assert info.value.code == 2


# --- opt-in real benchmark -----------------------------------------------------------------------


@pytest.mark.model
def test_real_cpu_smoke_benchmark(tmp_path: Path) -> None:
    """Tiny real benchmark of both backends on CPU: checks the result files, not the speed."""
    import os

    if os.environ.get("EDGEVISION_RUN_MODEL_TESTS") != "1":
        pytest.skip("real benchmark is opt-in: set EDGEVISION_RUN_MODEL_TESTS=1 (needs the vision, detector and "
                    "onnxruntime extras, models/yolov8n.pt and models/yolov8n.onnx)")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("ultralytics")
    from ultralytics.utils import ASSETS

    for model in ("yolov8n.pt", "yolov8n.onnx"):
        if not (REPO / "models" / model).is_file():
            pytest.skip(f"models/{model} not found")
    results = {}
    for backend, model in (("pytorch", "yolov8n.pt"), ("onnx", "yolov8n.onnx")):
        out = tmp_path / f"{backend}.json"
        result = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "benchmark.py"), str(ASSETS / "bus.jpg"), "--backend", backend,
             "--model", str(REPO / "models" / model), "--device", "cpu", "--warmup", "2", "--iterations", "3",
             "--runs", "1", "--output", str(out), "--config", str(CONFIG), "--quiet"],
            capture_output=True, text=True, check=False, cwd=REPO,
        )
        assert result.returncode == 0, result.stderr
        results[backend] = json.loads(out.read_text(encoding="utf-8"))
    for doc in results.values():
        assert doc["backend"]["device"] == "cpu" and doc["summary"]["latency_ms_pooled"]["detect"]["count"] == 3
        assert {"preprocess_ms", "inference_ms", "postprocess_ms"} <= set(doc["summary"]["latency_ms_pooled"])
    assert results["onnx"]["backend"]["execution_providers"] == ["CPUExecutionProvider"]
    # Same square input tensor, so both backends found the same objects.
    assert results["pytorch"]["summary"]["detections_total_per_run"] == results["onnx"]["summary"]["detections_total_per_run"]
