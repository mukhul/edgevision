"""Tests for edgevision.benchmarks.report: schema, metadata, writing, summaries, comparison.

The run results here are hand-made, not measurements.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from edgevision.benchmarks import report
from edgevision.benchmarks.report import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    build_report,
    comparability_problems,
    display_path,
    file_identity,
    format_comparison,
    format_summary,
    load_report,
    write_report,
)
from edgevision.benchmarks.runner import BenchmarkPlan, RunResult


def make_run(run: int, detect: list[float], init_s: float = 1.5, stages: bool = True) -> RunResult:
    samples = {"detect": detect}
    if stages:
        samples |= {"preprocess_ms": [1.0] * len(detect), "inference_ms": [d - 2 for d in detect],
                    "postprocess_ms": [1.0] * len(detect)}
    return RunResult(run=run, init_s=init_s, first_inference_ms=250.0, warmup_iterations=2, warmup_s=0.5,
                     measured_iterations=len(detect), measured_wall_s=sum(detect) / 1000 + 0.001, samples_ms=samples,
                     detections_total=10, device="cpu", execution_providers=["CPUExecutionProvider"])


def make_doc(**changes) -> dict:
    plan = BenchmarkPlan(warmup=2, iterations=4, runs=2).to_dict()
    backend = {"name": "onnx", "runtime": "ONNX Runtime 1.31.0, providers: CPUExecutionProvider",
               "model": {"path": "models/yolov8n.onnx", "size_bytes": 3, "sha256": "ab"}, "requested_device": "cpu",
               "device": "cpu", "execution_providers": ["CPUExecutionProvider"], "letterbox": "square",
               "model_input_shape": [1, 3, 640, 640], "precision": "fp32",
               "settings": {"input_size": 640, "conf_threshold": 0.25, "iou_threshold": 0.7, "classes": None}}
    input_info = {"kind": "video", "file": {"path": "data/input/sample.mp4", "size_bytes": 9, "sha256": "cd"},
                  "width": 1024, "height": 576, "fps": 30.0, "frame_count_reported": 113, "frames_in_memory": 32}
    env = {"os": {"platform": "Windows-11"}, "cpu": {"model": "AMD Ryzen 7 5700U"}, "python": {"version": "3.11.9"}}
    doc = build_report(plan=plan, backend=backend, input_info=input_info,
                       runs=[make_run(1, [10.0, 12.0, 11.0, 30.0]), make_run(2, [11.0, 11.0, 13.0, 12.0], init_s=0.5)],
                       library_import={"modules": ["onnxruntime"], "seconds": 0.2, "already_imported": False,
                                       "error": None},
                       environment=env, git={"commit": "abc", "uncommitted_changes": False}, warnings=[])
    for path, value in changes.items():
        target = doc
        keys = path.split(".")
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    return doc


# --- document --------------------------------------------------------------------------------


def test_document_schema() -> None:
    doc = make_doc()
    assert doc["schema"] == SCHEMA_NAME and doc["schema_version"] == SCHEMA_VERSION
    assert list(doc) == ["schema", "schema_version", "edgevision_version", "created_utc", "git", "benchmark",
                         "backend", "input", "environment", "library_import", "summary", "runs",
                         "timing_definitions", "limitations", "warnings"]
    run = doc["runs"][0]
    assert run["latency_ms"]["detect"]["count"] == 4 and run["samples_ms"]["detect"] == [10.0, 12.0, 11.0, 30.0]
    assert run["throughput_fps"] == pytest.approx(4 / 0.064)
    for key in ("init_s", "first_inference_ms", "warmup_s", "measured_wall_s", "device", "execution_providers"):
        assert key in run
    for key in ("latency_ms.detect", "init_s", "throughput_fps", "first_inference_ms", "statistics"):
        assert key in doc["timing_definitions"]


def test_summary_pools_runs_and_shows_their_spread() -> None:
    s = make_doc()["summary"]
    assert s["latency_ms_pooled"]["detect"]["count"] == 8
    assert s["latency_ms_pooled"]["detect"]["max"] == 30.0
    assert s["detect_median_ms_per_run"] == [11.5, 11.5]
    assert s["init_s_per_run"] == [1.5, 0.5]
    assert len(s["throughput_fps_per_run"]) == 2
    assert s["throughput_fps"]["min"] <= s["throughput_fps"]["mean"] <= s["throughput_fps"]["max"]


def test_stage_stats_only_when_every_run_has_them() -> None:
    doc = build_report(plan={}, backend={}, input_info={}, runs=[make_run(1, [1.0, 2.0]),
                       make_run(2, [1.0, 2.0], stages=False)], library_import={}, environment={}, git=None, warnings=[])
    assert set(doc["summary"]["latency_ms_pooled"]) == {"detect"}


def test_report_needs_runs() -> None:
    with pytest.raises(ValueError, match="at least one run"):
        build_report(plan={}, backend={}, input_info={}, runs=[], library_import={}, environment={}, git=None,
                     warnings=[])


def test_json_round_trip_keeps_full_precision(tmp_path: Path) -> None:
    doc = make_doc()
    doc["runs"][0]["samples_ms"]["detect"][0] = 1 / 3
    path = tmp_path / "deep" / "folder" / "result.json"
    write_report(path, doc)
    loaded = load_report(path)
    assert loaded == doc and loaded["runs"][0]["samples_ms"]["detect"][0] == 1 / 3
    assert not list(path.parent.glob("*.partial"))


# --- writing ---------------------------------------------------------------------------------


def test_existing_file_is_not_replaced_without_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text("previous", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--overwrite"):
        write_report(path, make_doc())
    assert path.read_text(encoding="utf-8") == "previous"
    write_report(path, make_doc(), overwrite=True)
    assert load_report(path)["schema"] == SCHEMA_NAME


def test_failed_write_leaves_nothing(tmp_path: Path) -> None:
    doc = make_doc()
    doc["runs"][0]["init_s"] = float("nan")  # not valid JSON
    path = tmp_path / "result.json"
    with pytest.raises(ValueError):
        write_report(path, doc)
    assert list(tmp_path.iterdir()) == []


def test_output_must_be_json(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=".json"):
        write_report(tmp_path / "result.txt", make_doc())


@pytest.mark.parametrize(
    ("content", "message"),
    [({"schema": "edgevision.video_detections", "schema_version": 1}, "not an EdgeVision benchmark"),
     ({"schema": SCHEMA_NAME, "schema_version": 99}, "schema version 99")],
)
def test_load_rejects_other_files(tmp_path: Path, content: dict, message: str) -> None:
    path = tmp_path / "other.json"
    path.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_report(path)


# --- metadata --------------------------------------------------------------------------------


def test_paths_are_relative_or_just_a_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    (project / "models").mkdir(parents=True)
    model = project / "models" / "m.onnx"
    model.write_bytes(b"abc")
    outside = tmp_path / "secret_user" / "video.mp4"
    outside.parent.mkdir()
    outside.write_bytes(b"x")
    monkeypatch.chdir(project)
    assert display_path(model) == "models/m.onnx"
    assert display_path("models/m.onnx") == "models/m.onnx"
    assert display_path(outside) == "video.mp4"  # no absolute path, no user name
    identity = file_identity(model)
    assert identity == {"path": "models/m.onnx", "size_bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}


def test_environment_snapshot_has_no_personal_paths() -> None:
    env = report.environment_snapshot("onnx")
    text = json.dumps(env)
    assert "executable" not in text and str(Path.home()) not in text
    assert {"os", "python", "cpu", "nvidia_gpus", "packages", "onnxruntime"} <= set(env)
    assert "torch" not in env


def test_onnx_environment_does_not_import_torch() -> None:
    code = ("import sys\nsys.modules['torch'] = None\nfrom edgevision.benchmarks.report import environment_snapshot\n"
            "env = environment_snapshot('onnx')\nprint('torch' in env)")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0 and result.stdout.strip() == "False", result.stderr


def test_git_revision_without_git(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(report.subprocess, "run", no_git)
    assert report.git_revision() is None


# --- human-readable output -------------------------------------------------------------------


def test_summary_text() -> None:
    doc = make_doc(warnings=["only 4 measured iterations per run: p90/p95 are not reliable"])
    text = format_summary(doc)
    for expected in ("onnx (providers CPUExecutionProvider)", "input tensor [1, 3, 640, 640]", "AMD Ryzen 7 5700U",
                     "Initialisation     1.500, 0.500 s", "First inference", "detect", "inference_ms", "Throughput",
                     "not accuracy", "p90/p95 are not reliable"):
        assert expected in text


def test_comparison_lists_differences_without_ranking() -> None:
    a = make_doc()
    b = make_doc(**{"backend.name": "pytorch", "backend.execution_providers": None,
                    "backend.model_input_shape": [1, 3, 384, 640], "benchmark.measured_iterations": 300})
    problems = comparability_problems([a, b])
    assert any(p.startswith("model input tensor differs") for p in problems)
    assert any(p.startswith("measured iterations differs") for p in problems)
    text = format_comparison([a, b], ["onnx_run", "pytorch_run"])
    assert "onnx_run" in text and "pytorch_run" in text and "Not directly comparable" in text
    assert "384x640" in text and "CPUExecutionProvider" in text
    for word in ("faster", "slower", "winner", "best"):
        assert word not in text.lower()


def test_identical_settings_are_reported_as_comparable() -> None:
    a = make_doc()
    b = copy.deepcopy(a)
    assert comparability_problems([a, b]) == []
    assert "Same input, settings" in format_comparison([a, b], ["a", "b"])


# --- interrupted measurements (e.g. the computer went to sleep) -----------------------------------


def test_suspect_iterations_are_flagged_not_removed() -> None:
    samples = [95.0] * 50
    samples[10] = 13_115_822.3  # a laptop asleep for 3.6 hours in the middle of an iteration
    samples[20] = 300.0  # ordinary jitter: 3x the median, never flagged
    suspects = report.find_suspect_iterations(samples)
    assert [s["iteration"] for s in suspects] == [10]
    assert suspects[0]["ms"] == 13_115_822.3 and suspects[0]["times_median"] == pytest.approx(138_061.3, rel=1e-3)
    assert report.find_suspect_iterations([5.0] * 9 + [900.0]) == []  # 180x the median but under 1 s: not flagged
    assert report.find_suspect_iterations([]) == []


def test_report_warns_about_interrupted_runs() -> None:
    clean = make_run(1, [95.0] * 20)
    disturbed = make_run(2, [95.0] * 19 + [2_602_072.3])
    doc = build_report(plan=BenchmarkPlan(warmup=2, iterations=20, runs=2).to_dict(), backend=make_doc()["backend"],
                       input_info=make_doc()["input"], runs=[clean, disturbed], library_import=make_doc()["library_import"],
                       environment=make_doc()["environment"], git=None, warnings=[])
    assert doc["runs"][0]["suspect_iterations"] == []
    assert [s["iteration"] for s in doc["runs"][1]["suspect_iterations"]] == [19]
    assert doc["runs"][1]["samples_ms"]["detect"][-1] == 2_602_072.3  # the data itself is untouched
    assert len(doc["warnings"]) == 1 and "run 2" in doc["warnings"][0] and "2602.1 s" in doc["warnings"][0]
    assert "WARNING: run 2" in format_summary(doc)
    text = format_comparison([doc, make_doc()], ["onnx_cpu", "other"])
    assert "WARNING: onnx_cpu is not reliable: run(s) 2" in text and "other is not reliable" not in text


def test_old_files_without_flags_are_checked_from_their_samples() -> None:
    doc = make_doc()
    for r in doc["runs"]:
        del r["suspect_iterations"]
    assert report.suspect_runs(doc) == []
    doc["runs"][1]["samples_ms"]["detect"][2] = 6_228.7
    assert report.suspect_runs(doc) == [2]
