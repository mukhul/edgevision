"""Tests for scripts/detect_image.py.

Argument errors and --help run the script as a real subprocess. The other
tests import the script and call main() in-process with the fake YOLO model
from conftest.py, so they never download weights or run a real network.
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
SCRIPT = REPO / "scripts" / "detect_image.py"
CONFIG = REPO / "configs" / "default.yaml"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("detect_image_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_subprocess(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False)


@pytest.fixture
def image_file(tmp_path: Path) -> Path:
    cv2 = pytest.importorskip("cv2", reason='OpenCV not installed (pip install -e ".[vision]")')
    np = pytest.importorskip("numpy")
    path = tmp_path / "photo.png"
    assert cv2.imwrite(str(path), np.zeros((120, 160, 3), dtype=np.uint8))
    return path


# --- command-line interface ------------------------------------------------------


def test_help_describes_the_options() -> None:
    result = run_subprocess("--help")
    assert result.returncode == 0
    for option in ("--model", "--device", "--conf", "--iou", "--imgsz", "--json"):
        assert option in result.stdout


@pytest.mark.parametrize(
    "args",
    [
        ["img.jpg", "--conf", "1.5"],
        ["img.jpg", "--iou", "abc"],
        ["img.jpg", "--device", "gpu"],
        ["img.jpg", "--imgsz", "0"],
        [],
    ],
)
def test_invalid_arguments_exit_with_status_2(args: list[str]) -> None:
    result = run_subprocess(*args)
    assert result.returncode == 2
    assert "usage:" in result.stderr


def test_input_size_must_be_a_multiple_of_32(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        load_script().main(["img.jpg", "--imgsz", "641", "--config", str(CONFIG)])
    assert info.value.code == 2
    assert "multiple of 32" in capsys.readouterr().err


# --- input errors ------------------------------------------------------------------


def test_missing_image(fake_yolo, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = load_script().main([str(tmp_path / "missing.jpg"), "--config", str(CONFIG)])
    assert code == 1
    assert "Image not found" in capsys.readouterr().err
    assert fake_yolo.loaded_paths == []  # no model was loaded for nothing


def test_file_that_is_not_an_image(fake_yolo, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("cv2", reason='OpenCV not installed (pip install -e ".[vision]")')
    path = tmp_path / "notes.jpg"
    path.write_text("not an image", encoding="utf-8")
    assert load_script().main([str(path), "--config", str(CONFIG)]) == 1
    assert "Could not decode" in capsys.readouterr().err


def test_missing_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = load_script().main(["img.jpg", "--config", str(tmp_path / "nope.yaml")])
    assert code == 1
    assert "Configuration file not found" in capsys.readouterr().err


# --- running the (fake) detector ------------------------------------------------------


def test_prints_detections(fake_yolo, image_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake_yolo.boxes = [(10, 20, 60, 90, 0.42, 0), (5, 5, 150, 110, 0.91, 15)]

    assert load_script().main([str(image_file), "--device", "cpu", "--config", str(CONFIG)]) == 0

    out = capsys.readouterr().out
    assert "(160 x 120 px)" in out
    assert "device: cpu" in out
    assert "Detections   2" in out
    lines = out.splitlines()
    cat_line = next(i for i, line in enumerate(lines) if " cat " in line)
    person_line = next(i for i, line in enumerate(lines) if " person " in line)
    assert cat_line < person_line  # sorted by confidence, highest first


def test_no_detections(fake_yolo, image_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert load_script().main([str(image_file), "--config", str(CONFIG)]) == 0
    assert "Detections   0" in capsys.readouterr().out


def test_defaults_come_from_config_and_options_override_them(fake_yolo, image_file: Path, tmp_path: Path) -> None:
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data["model"].update(input_size=320, conf_threshold=0.3, classes=[15], weights="models/custom.pt")
    config = tmp_path / "custom.yaml"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")

    script = load_script()
    assert script.main([str(image_file), "--config", str(config)]) == 0
    assert script.main([str(image_file), "--config", str(config), "--conf", "0.6", "--imgsz", "640"]) == 0

    from_config, overridden = fake_yolo.predict_calls
    assert (from_config["imgsz"], from_config["conf"], from_config["classes"]) == (320, 0.3, [15])
    assert (overridden["imgsz"], overridden["conf"]) == (640, 0.6)
    assert fake_yolo.loaded_paths[0] == "models/custom.pt"


def test_writes_json(fake_yolo, image_file: Path, tmp_path: Path) -> None:
    fake_yolo.boxes = [(5, 5, 150, 110, 0.91, 15)]
    out = tmp_path / "results" / "photo.json"  # parent folder is created

    assert load_script().main([str(image_file), "--json", str(out), "--config", str(CONFIG)]) == 0

    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["backend"] == "pytorch"
    assert doc["image_size"] == {"width": 160, "height": 120}
    assert doc["settings"] == {"input_size": 640, "conf_threshold": 0.25, "iou_threshold": 0.7, "classes": None}
    assert doc["detections"] == [
        {"frame_index": 0, "bbox_xyxy": [5.0, 5.0, 150.0, 110.0], "confidence": 0.91,
         "class_id": 15, "class_name": "cat"}
    ]


def test_existing_json_is_not_overwritten_without_permission(
    fake_yolo, image_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "photo.json"
    out.write_text("previous results", encoding="utf-8")
    script = load_script()

    assert script.main([str(image_file), "--json", str(out), "--config", str(CONFIG)]) == 1
    assert "already exists" in capsys.readouterr().err
    assert out.read_text(encoding="utf-8") == "previous results"

    assert script.main([str(image_file), "--json", str(out), "--overwrite", "--config", str(CONFIG)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["backend"] == "pytorch"


def test_cuda_request_without_cuda_fails(fake_yolo, image_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = load_script().main([str(image_file), "--device", "cuda", "--config", str(CONFIG)])
    assert code == 1
    assert "CUDA is not available" in capsys.readouterr().err


def test_model_load_failure(fake_yolo, image_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake_yolo.load_error = RuntimeError("invalid load key")
    code = load_script().main([str(image_file), "--model", "broken.pt", "--config", str(CONFIG)])
    assert code == 1
    err = capsys.readouterr().err
    assert "Could not load YOLO model 'broken.pt'" in err and "invalid load key" in err


def test_inference_failure(fake_yolo, image_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake_yolo.predict_error = RuntimeError("kernel crashed")
    assert load_script().main([str(image_file), "--config", str(CONFIG)]) == 1
    assert "inference failed" in capsys.readouterr().err
