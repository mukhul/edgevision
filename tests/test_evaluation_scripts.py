"""Tests for scripts/evaluate.py and scripts/compare_detections.py.

Everything is synthetic: a hand-written COCO file, blank images, a fake
detector with fixed outputs, and hand-written detection files. The numbers
test the commands, not any model.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("numpy")

from edgevision.models.base import Detection  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "default.yaml"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"{name}_eval_script", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(name: str, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(REPO / "scripts" / f"{name}.py"), *args], capture_output=True,
                          text=True, check=False, cwd=cwd)


# --- evaluate.py -----------------------------------------------------------------------------------


class FakeDetector:
    class_names = {0: "person", 2: "car"}
    device, description, backend_name = "cpu", "fake runtime", "pytorch"

    def __init__(self, outputs: dict[int, list]) -> None:
        self.outputs = outputs
        self.calls: list[tuple[tuple[int, ...], int]] = []

    def detect(self, frame, frame_index=0):
        self.calls.append((frame.shape, frame_index))
        return [Detection(frame_index, box, conf, cls, self.class_names[cls]) for box, conf, cls in
                self.outputs.get(frame_index, [])]

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@pytest.fixture
def dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cv2 = pytest.importorskip("cv2", reason='OpenCV not installed (pip install -e ".[vision]")')
    import numpy as np

    (tmp_path / "images").mkdir()
    for name, (w, h) in {"a.jpg": (64, 48), "b.jpg": (32, 32)}.items():
        assert cv2.imwrite(str(tmp_path / "images" / name), np.zeros((h, w, 3), np.uint8))
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 64, "height": 48},
                   {"id": 2, "file_name": "b.jpg", "width": 32, "height": 32}],
        "categories": [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}, {"id": 50, "name": "unicorn"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
            {"id": 2, "image_id": 1, "category_id": 3, "bbox": [20, 20, 10, 10]},
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [5, 5, 10, 10]},
            {"id": 4, "image_id": 2, "category_id": 50, "bbox": [1, 1, 3, 3]},
        ],
    }
    (tmp_path / "ann.json").write_text(json.dumps(coco), encoding="utf-8")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "yolov8n.pt").write_bytes(b"fake")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def evaluate_with(script: ModuleType, outputs: dict, monkeypatch: pytest.MonkeyPatch) -> FakeDetector:
    fake = FakeDetector(outputs)
    monkeypatch.setattr(script, "create_detector", lambda *a, **k: fake)
    return fake


ARGS = ["--annotations", "ann.json", "--images", "images", "--config", str(CONFIG)]


def test_evaluate_writes_metrics(dataset: Path, monkeypatch, capsys) -> None:
    script = load("evaluate")
    fake = evaluate_with(script, {
        0: [((0, 0, 10, 10), 0.9, 0), ((40, 0, 50, 10), 0.8, 0)],  # one correct person, one false alarm
        1: [((5, 5, 15, 15), 0.7, 0)],  # correct person; the car in image 0 is missed
    }, monkeypatch)
    assert script.main([*ARGS, "--output", "eval.json"]) == 0
    assert [shape for shape, _ in fake.calls] == [(48, 64, 3), (32, 32, 3)]
    doc = json.loads((dataset / "eval.json").read_text(encoding="utf-8"))
    assert doc["schema"] == "edgevision.evaluation" and doc["schema_version"] == 1
    assert doc["dataset"]["images"] == 2 and doc["dataset"]["objects"] == 3
    assert doc["dataset"]["unmapped_categories"] == ["unicorn"]
    m = doc["metrics"]
    person, car = m["per_class"]["person"], m["per_class"]["car"]
    assert (person["tp"], person["fp"], person["fn"]) == (2, 1, 0)
    assert person["precision"] == pytest.approx(2 / 3) and person["recall"] == 1.0
    assert (car["tp"], car["fn"], car["precision"], car["recall"], car["ap50"]) == (0, 1, None, 0.0, 0.0)
    assert m["overall"]["recall"] == pytest.approx(2 / 3)
    out = capsys.readouterr().out
    assert "Detection accuracy on the supplied labelled dataset" in out and "mAP@0.5" in out
    assert "unicorn" in out and "--conf 0.001" in out


def test_evaluate_rejects_mismatched_image_sizes(dataset: Path, monkeypatch, capsys) -> None:
    data = json.loads((dataset / "ann.json").read_text(encoding="utf-8"))
    data["images"][0]["width"] = 640
    (dataset / "ann.json").write_text(json.dumps(data), encoding="utf-8")
    script = load("evaluate")
    evaluate_with(script, {}, monkeypatch)
    assert script.main([*ARGS, "--output", "eval.json"]) == 1
    assert "boxes would not line up" in capsys.readouterr().err
    assert not (dataset / "eval.json").exists()


def test_evaluate_missing_inputs(dataset: Path, monkeypatch, capsys) -> None:
    script = load("evaluate")
    evaluate_with(script, {}, monkeypatch)
    assert script.main(["--annotations", "nope.json", "--images", "images", "--config", str(CONFIG)]) == 1
    assert "Annotation file not found" in capsys.readouterr().err
    assert script.main(["--annotations", "ann.json", "--images", "nope", "--config", str(CONFIG)]) == 1
    assert "Images folder not found" in capsys.readouterr().err
    (dataset / "images" / "b.jpg").unlink()
    assert script.main([*ARGS, "--output", "eval.json"]) == 1
    assert "not found" in capsys.readouterr().err
    (dataset / "eval.json").write_text("keep", encoding="utf-8")
    assert script.main([*ARGS, "--output", "eval.json"]) == 1
    assert "already exists" in capsys.readouterr().err


def test_evaluate_help_and_arguments(tmp_path: Path) -> None:
    result = run("evaluate", "--help", cwd=tmp_path)
    assert result.returncode == 0 and "--annotations" in result.stdout and list(tmp_path.iterdir()) == []
    assert run("evaluate", "--images", "x").returncode == 2  # --annotations is required
    result = run("evaluate", "--annotations", "a.json", "--images", "x", "--backend", "onnx", "--model", "m.pt")
    assert result.returncode == 2 and "needs an .onnx file" in result.stderr


# --- compare_detections.py -------------------------------------------------------------------------


def video_json(path: Path, backend: str, frames: list[list[tuple]], **detector) -> Path:
    doc = {
        "schema": "edgevision.video_detections", "schema_version": 1,
        "video": {"path": "data/input/sample.mp4", "width": 1024, "height": 576},
        "detector": {"backend": backend, "model": f"models/m.{backend}", "device": "cpu",
                     "settings": {"input_size": 640, "conf_threshold": 0.25, "iou_threshold": 0.7, "classes": None},
                     **detector},
        "frames": [{"frame_index": i, "detections": [
            {"bbox_xyxy": list(box), "confidence": conf, "class_id": cls, "class_name": name}
            for box, conf, cls, name in dets]} for i, dets in enumerate(frames)],
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_compare_detections(tmp_path: Path, capsys) -> None:
    a = video_json(tmp_path / "a.json", "pytorch", [[((0, 0, 10, 10), 0.9, 0, "person")], []])
    b = video_json(tmp_path / "b.json", "onnx", [[((0, 0, 10, 10), 0.85, 0, "person")], [((5, 5, 9, 9), 0.3, 15, "cat")]])
    script = load("compare_detections")
    assert script.main([str(a), str(b), "--output", str(tmp_path / "out" / "agreement.json")]) == 0
    out = capsys.readouterr().out
    assert "NOT an accuracy measure" in out and "Matched pairs          1" in out
    assert "Only in A / only in B  0 / 1" in out and "pads frames only to a multiple of 32" in out
    saved = json.loads((tmp_path / "out" / "agreement.json").read_text(encoding="utf-8"))
    assert saved["schema"] == "edgevision.detection_agreement" and saved["agreement"]["only_b"] == 1


def test_compare_detections_errors(tmp_path: Path, capsys) -> None:
    script = load("compare_detections")
    a = video_json(tmp_path / "a.json", "onnx", [[], []])
    short = video_json(tmp_path / "short.json", "onnx", [[]])
    assert script.main([str(a), str(short)]) == 1
    assert "different number of frames" in capsys.readouterr().err
    other = tmp_path / "bench.json"
    other.write_text(json.dumps({"schema": "edgevision.benchmark", "schema_version": 1}), encoding="utf-8")
    assert script.main([str(a), str(other)]) == 1
    assert "not a detect_video.py --json result" in capsys.readouterr().err
    assert script.main([str(a), str(tmp_path / "missing.json")]) == 1
    broken = video_json(tmp_path / "broken.json", "onnx", [[((10, 0, 0, 10), 0.5, 0, "person")], []])
    assert script.main([str(a), str(broken)]) == 1
    assert "invalid detection in frame 0" in capsys.readouterr().err


def test_compare_detections_flags_different_settings(tmp_path: Path, capsys) -> None:
    a = video_json(tmp_path / "a.json", "onnx", [[]])
    b = video_json(tmp_path / "b.json", "onnx", [[]], settings={"input_size": 320, "conf_threshold": 0.25,
                                                                "iou_threshold": 0.7, "classes": None})
    assert load("compare_detections").main([str(a), str(b)]) == 0
    assert "setting input_size differs: 640 vs 320" in capsys.readouterr().out
