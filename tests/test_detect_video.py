"""Tests for scripts/detect_video.py.

Argument errors and --help run the script as a subprocess. Other tests call
main() in-process with the fake YOLO model from conftest.py and a small
synthetic video (needs OpenCV), so nothing is downloaded and no real model runs.
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
SCRIPT = REPO / "scripts" / "detect_video.py"
CONFIG = REPO / "configs" / "default.yaml"
FRAMES_IN_VIDEO = 12


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("detect_video_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_subprocess(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False)


@pytest.fixture
def video(tmp_path: Path) -> Path:
    cv2 = pytest.importorskip("cv2", reason='OpenCV not installed (pip install -e ".[vision]")')
    np = pytest.importorskip("numpy")
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 120))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot encode MJPG video")
    try:
        for _ in range(FRAMES_IN_VIDEO):
            writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    finally:
        writer.release()
    return path


def custom_config(tmp_path: Path, **video_section: object) -> Path:
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data["video"].update(video_section)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# --- command-line interface ------------------------------------------------------------


def test_help_describes_the_options() -> None:
    result = run_subprocess("--help")
    assert result.returncode == 0
    for option in ("--max-frames", "--all-frames", "--model", "--device", "--conf", "--iou", "--json", "--overwrite"):
        assert option in result.stdout


@pytest.mark.parametrize(
    "args",
    [
        ["clip.mp4", "--max-frames", "0"],
        ["clip.mp4", "--max-frames", "5", "--all-frames"],  # mutually exclusive
        ["clip.mp4", "--conf", "2"],
        ["clip.mp4", "--device", "gpu"],
    ],
)
def test_invalid_arguments_exit_with_status_2(args: list[str]) -> None:
    result = run_subprocess(*args)
    assert result.returncode == 2
    assert "usage:" in result.stderr


def test_missing_video_fails_before_loading_the_model(
    fake_yolo, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = load_script().main([str(tmp_path / "missing.mp4"), "--config", str(CONFIG)])
    assert code == 1
    assert "Video not found" in capsys.readouterr().err
    assert fake_yolo.loaded_paths == []


# --- processing ------------------------------------------------------------------------


def test_processes_video_and_prints_summary(fake_yolo, video: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake_yolo.boxes = [(10, 10, 50, 60, 0.9, 15), (60, 10, 100, 80, 0.6, 0)]

    code = load_script().main([str(video), "--device", "cpu", "--config", str(CONFIG)])

    assert code == 0
    assert len(fake_yolo.loaded_paths) == 1  # the model is loaded once, not per frame
    assert len(fake_yolo.predict_calls) == FRAMES_IN_VIDEO
    out = " ".join(capsys.readouterr().out.split())  # ignore column alignment
    assert f"Frames processed {FRAMES_IN_VIDEO}" in out
    assert f"Total detections {2 * FRAMES_IN_VIDEO}" in out
    assert "not a benchmark" in out


def test_config_frame_limit_applies_by_default(fake_yolo, video: Path, tmp_path: Path) -> None:
    config = custom_config(tmp_path, max_frames=5)
    assert load_script().main([str(video), "--config", str(config)]) == 0
    assert len(fake_yolo.predict_calls) == 5


def test_max_frames_and_all_frames_override_the_config(fake_yolo, video: Path, tmp_path: Path) -> None:
    config = custom_config(tmp_path, max_frames=5)
    script = load_script()
    assert script.main([str(video), "--config", str(config), "--max-frames", "3"]) == 0
    assert len(fake_yolo.predict_calls) == 3
    assert script.main([str(video), "--config", str(config), "--all-frames"]) == 0
    assert len(fake_yolo.predict_calls) == 3 + FRAMES_IN_VIDEO


def test_video_path_defaults_to_config(fake_yolo, video: Path, tmp_path: Path) -> None:
    config = custom_config(tmp_path, input=str(video))
    assert load_script().main(["--config", str(config)]) == 0
    assert len(fake_yolo.predict_calls) == FRAMES_IN_VIDEO


# --- JSON output -------------------------------------------------------------------------


def test_writes_json_matching_the_summary(fake_yolo, video: Path, tmp_path: Path) -> None:
    fake_yolo.boxes = [(10, 10, 50, 60, 0.9, 15)]
    out = tmp_path / "results" / "clip.json"

    assert load_script().main([str(video), "--max-frames", "4", "--json", str(out), "--config", str(CONFIG)]) == 0

    doc = json.loads(out.read_text(encoding="utf-8"))  # Python's standard JSON parser
    assert doc["schema"] == "edgevision.video_detections"
    assert doc["video"]["width"] == 160 and doc["video"]["height"] == 120
    assert doc["detector"]["device"] == "cpu" and doc["detector"]["settings"]["conf_threshold"] == 0.25
    assert [f["frame_index"] for f in doc["frames"]] == [0, 1, 2, 3]
    assert doc["frames"][0]["detections"] == [
        {"bbox_xyxy": [10.0, 10.0, 50.0, 60.0], "confidence": 0.9, "class_id": 15, "class_name": "cat"}
    ]
    s = doc["summary"]
    assert s["frames_processed"] == 4 and s["total_detections"] == 4
    assert s["detections_by_class"] == {"cat": 4}
    assert s["stop_reason"] == "max_frames" and s["completed"] is True


def test_existing_json_is_not_overwritten_without_permission(
    fake_yolo, video: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "clip.json"
    out.write_text("previous results", encoding="utf-8")
    script = load_script()

    assert script.main([str(video), "--json", str(out), "--config", str(CONFIG)]) == 1
    assert "already exists" in capsys.readouterr().err
    assert out.read_text(encoding="utf-8") == "previous results"
    assert fake_yolo.loaded_paths == []  # refused before doing any work

    assert script.main([str(video), "--json", str(out), "--overwrite", "--config", str(CONFIG)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["summary"]["frames_processed"] == FRAMES_IN_VIDEO


def test_inference_failure_leaves_no_results_file(
    fake_yolo, video: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_yolo.predict_error = RuntimeError("kernel crashed")
    out = tmp_path / "clip.json"

    assert load_script().main([str(video), "--json", str(out), "--config", str(CONFIG)]) == 1

    err = capsys.readouterr().err
    assert "Detection failed on frame 0" in err and "kernel crashed" in err
    assert "No results file was written" in err
    assert not out.exists()
    assert not out.with_name(out.name + ".partial").exists()


def test_ctrl_c_leaves_no_results_file(
    fake_yolo, video: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_yolo.predict_error = KeyboardInterrupt()  # what Ctrl+C raises mid-inference
    out = tmp_path / "clip.json"

    assert load_script().main([str(video), "--json", str(out), "--config", str(CONFIG)]) == 130

    assert "Interrupted" in capsys.readouterr().err
    assert not out.exists()
    assert not out.with_name(out.name + ".partial").exists()


def test_cuda_request_without_cuda_fails(fake_yolo, video: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert load_script().main([str(video), "--device", "cuda", "--config", str(CONFIG)]) == 1
    assert "CUDA is not available" in capsys.readouterr().err


def test_undecodable_video_fails_clearly(fake_yolo, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("cv2", reason='OpenCV not installed (pip install -e ".[vision]")')
    broken = tmp_path / "broken.mp4"
    broken.write_text("not a video", encoding="utf-8")
    assert load_script().main([str(broken), "--config", str(CONFIG)]) == 1
    assert "ERROR:" in capsys.readouterr().err
