"""Choosing between the PyTorch and ONNX Runtime backends.

Covers the factory and the --backend option of both detection commands. The
commands run in-process with the fake PyTorch/Ultralytics (``fake_yolo``) and
the fake ONNX Runtime (``fake_ort``) from conftest.py: no real model runs.
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

from edgevision.models import OnnxYoloDetector, PytorchYoloDetector, create_detector
from edgevision.models.factory import check_model_matches_backend

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "default.yaml"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"{name}_script_b", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(REPO / "scripts" / f"{name}.py"), *args], capture_output=True, text=True, check=False)


@pytest.fixture
def cv2():
    return pytest.importorskip("cv2", reason='OpenCV not installed (pip install -e ".[vision]")')


@pytest.fixture
def onnx_file(tmp_path: Path) -> Path:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"fake")
    return path


@pytest.fixture
def image_file(tmp_path: Path, cv2) -> Path:
    import numpy as np

    path = tmp_path / "photo.png"
    assert cv2.imwrite(str(path), np.zeros((480, 640, 3), dtype=np.uint8))
    return path


@pytest.fixture
def video(tmp_path: Path, cv2) -> Path:
    import numpy as np

    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 120))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot encode MJPG video")
    try:
        for _ in range(6):
            writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    finally:
        writer.release()
    return path


def config_with_onnx_path(tmp_path: Path, onnx_path: Path) -> Path:
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data["model"]["onnx_path"] = onnx_path.as_posix()
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# --- factory ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("backend", "model", "message"),
    [
        ("pytorch", "models/yolov8n.onnx", "is an ONNX model; use --backend onnx"),
        ("pytorch", "MODELS/X.ONNX", "is an ONNX model"),
        ("onnx", "models/yolov8n.pt", "needs an .onnx file.*export_onnx.py"),
        ("onnx", "yolov8n", "needs an .onnx file"),
        ("tensorrt", "m.engine", "backend must be one of"),
    ],
)
def test_model_must_match_backend(backend: str, model: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        check_model_matches_backend(backend, model)


@pytest.mark.parametrize(("backend", "model"), [("pytorch", "yolov8n.pt"), ("pytorch", "custom"), ("onnx", "m.onnx")])
def test_matching_models_are_accepted(backend: str, model: str) -> None:
    check_model_matches_backend(backend, model)


def test_factory_creates_each_backend(fake_yolo, fake_ort, cv2, onnx_file: Path) -> None:
    assert isinstance(create_detector("pytorch", "models/yolov8n.pt", device="cpu"), PytorchYoloDetector)
    detector = create_detector("onnx", onnx_file, device="cpu")
    assert isinstance(detector, OnnxYoloDetector) and detector.providers == ("CPUExecutionProvider",)


# --- detect_image.py -------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["detect_image", "detect_video"])
def test_help_lists_backend_option(name: str) -> None:
    result = run(name, "--help")
    assert result.returncode == 0
    assert "--backend {pytorch,onnx}" in result.stdout and "model.onnx_path" in result.stdout


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--backend", "tensorrt"], "invalid choice"),
        (["--backend", "onnx", "--model", "models/yolov8n.pt"], "needs an .onnx file"),
        (["--model", "models/yolov8n.onnx"], "use --backend onnx"),
    ],
)
@pytest.mark.parametrize("name", ["detect_image", "detect_video"])
def test_backend_argument_errors_exit_with_status_2(name: str, args: list[str], message: str) -> None:
    result = run(name, "input.file", *args)
    assert result.returncode == 2 and message in result.stderr


def test_image_with_onnx_backend(fake_ort, image_file: Path, onnx_file: Path, tmp_path: Path, capsys, make_raw_output) -> None:
    fake_ort.output = make_raw_output([(320, 320, 100, 60, 2, 0.9)])
    out = tmp_path / "out.json"
    config = config_with_onnx_path(tmp_path, onnx_file)  # the default ONNX model comes from model.onnx_path
    code = load("detect_image").main([str(image_file), "--backend", "onnx", "--device", "cpu",
                                      "--json", str(out), "--config", str(config)])
    assert code == 0
    printed = capsys.readouterr()
    assert "ONNX Runtime 1.31.0, providers: CPUExecutionProvider, device: cpu" in printed.out
    assert "download" not in printed.err  # ONNX models are never downloaded
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["backend"] == "onnx" and doc["model"] == onnx_file.as_posix() and doc["device"] == "cpu"
    assert doc["execution_providers"] == ["CPUExecutionProvider"]
    assert [d["class_name"] for d in doc["detections"]] == ["cat"]
    assert doc["detections"][0]["bbox_xyxy"] == pytest.approx([270, 210, 370, 270])


def test_image_with_missing_onnx_model(fake_ort, image_file: Path, tmp_path: Path, capsys) -> None:
    missing = tmp_path / "nope.onnx"
    code = load("detect_image").main([str(image_file), "--backend", "onnx", "--model", str(missing), "--config", str(CONFIG)])
    assert code == 1
    err = capsys.readouterr().err
    assert "ONNX model not found" in err and "export_onnx.py" in err and "will try to download" not in err


def test_image_with_onnx_cuda_request_but_no_gpu(fake_ort, image_file: Path, onnx_file: Path, capsys) -> None:
    code = load("detect_image").main([str(image_file), "--backend", "onnx", "--model", str(onnx_file),
                                      "--device", "cuda", "--config", str(CONFIG)])
    assert code == 1 and "no CUDAExecutionProvider" in capsys.readouterr().err


def test_pytorch_stays_the_default(fake_yolo, image_file: Path, tmp_path: Path, capsys) -> None:
    out = tmp_path / "pt.json"
    assert load("detect_image").main([str(image_file), "--json", str(out), "--config", str(CONFIG)]) == 0
    assert fake_yolo.loaded_paths == ["models/yolov8n.pt"]
    assert "PyTorch via Ultralytics, device: cpu" in capsys.readouterr().out
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["backend"] == "pytorch" and "execution_providers" not in doc  # PyTorch JSON layout unchanged
    assert list(doc) == ["edgevision_version", "backend", "image", "image_size", "model", "device", "settings",
                         "bbox_format", "detections"]


# --- detect_video.py -------------------------------------------------------------------------


def test_video_with_onnx_backend_tracking_and_outputs(
    fake_ort, video: Path, onnx_file: Path, tmp_path: Path, fake_tracker_class, capsys, make_raw_output
) -> None:
    # 160 x 120 frames -> 640 x 640 input: scale 4, 80 rows of padding at the top.
    fake_ort.output = make_raw_output([(200, 300, 80, 40, 2, 0.9), (500, 400, 40, 80, 0, 0.6)])
    script = load("detect_video")
    script.ByteTrackTracker = lambda config=None: fake_tracker_class(config)
    out_json, out_video = tmp_path / "onnx.json", tmp_path / "onnx.mp4"
    code = script.main([str(video), "--backend", "onnx", "--model", str(onnx_file), "--device", "cpu", "--tracking",
                        "--json", str(out_json), "--output-video", str(out_video), "--config", str(CONFIG), "--quiet"])
    assert code == 0
    assert len(fake_ort.sessions) == 1 and len(fake_ort.feeds) == 6  # one session, one run per frame
    printed = " ".join(capsys.readouterr().out.split())
    assert "ONNX Runtime 1.31.0, providers: CPUExecutionProvider, device: cpu" in printed
    assert "Total detections 12" in printed

    doc = json.loads(out_json.read_text(encoding="utf-8"))
    assert doc["detector"]["backend"] == "onnx" and doc["detector"]["execution_providers"] == ["CPUExecutionProvider"]
    first = doc["frames"][0]["detections"]
    assert [(d["class_name"], d["track_id"]) for d in first] == [("cat", 1), ("person", 2)]
    assert first[0]["bbox_xyxy"] == pytest.approx([40, 50, 60, 60])  # (160..240, 280..320) / 4, minus 80 px padding

    cv2 = pytest.importorskip("cv2")
    capture = cv2.VideoCapture(str(out_video))
    frames = 0
    while capture.read()[0]:
        frames += 1
    capture.release()
    assert frames == 6
    assert doc["summary"]["total_detections"] == 12


def test_video_pytorch_json_has_no_provider_field(fake_yolo, video: Path, tmp_path: Path) -> None:
    out = tmp_path / "pt.json"
    assert load("detect_video").main([str(video), "--json", str(out), "--config", str(CONFIG), "--quiet"]) == 0
    detector = json.loads(out.read_text(encoding="utf-8"))["detector"]
    assert list(detector) == ["backend", "model", "device", "settings"] and detector["backend"] == "pytorch"


def test_video_with_onnx_input_size_mismatch(fake_ort, video: Path, onnx_file: Path, tmp_path: Path, capsys) -> None:
    fake_ort.input_shape = [1, 3, 320, 320]
    out = tmp_path / "x.json"
    code = load("detect_video").main([str(video), "--backend", "onnx", "--model", str(onnx_file), "--json", str(out),
                                      "--config", str(CONFIG), "--quiet"])
    assert code == 1
    err = capsys.readouterr().err
    assert "--imgsz 320" in err and "No output files were written" in err
    assert not out.exists()


# --- options used by benchmarks ----------------------------------------------------------------


def test_pytorch_letterbox_option(fake_yolo) -> None:
    import numpy as np

    frame = np.zeros((48, 64, 3), np.uint8)
    default = PytorchYoloDetector("models/yolov8n.pt", device="cpu")
    default.detect(frame)
    assert default.letterbox == "minimal" and "rect" not in fake_yolo.predict_calls[-1]  # unchanged default call
    square = create_detector("pytorch", "models/yolov8n.pt", device="cpu", letterbox="square")
    square.detect(frame)
    assert square.letterbox == "square" and fake_yolo.predict_calls[-1]["rect"] is False
    with pytest.raises(ValueError, match="letterbox must be one of"):
        PytorchYoloDetector("models/yolov8n.pt", letterbox="stretch")


def test_pytorch_stage_timings_come_from_ultralytics(fake_yolo, monkeypatch) -> None:
    import numpy as np

    detector = PytorchYoloDetector("models/yolov8n.pt", device="cpu")
    assert detector.last_timings is None
    detector.detect(np.zeros((48, 64, 3), np.uint8))
    assert detector.last_timings is None  # the fake result reports no speed: nothing is invented
    original = fake_yolo.make_result

    def with_speed():
        result = original()
        result.speed = {"preprocess": 1.5, "inference": 20.0, "postprocess": 0.5}
        return result

    monkeypatch.setattr(fake_yolo, "make_result", with_speed)
    detector.detect(np.zeros((48, 64, 3), np.uint8))
    assert detector.last_timings == {"preprocess_ms": 1.5, "inference_ms": 20.0, "postprocess_ms": 0.5}


def test_onnx_stage_timings(fake_ort, cv2, onnx_file: Path) -> None:
    import numpy as np

    detector = create_detector("onnx", onnx_file, device="cpu", letterbox="square")
    assert detector.last_timings is None
    detector.detect(np.zeros((48, 64, 3), np.uint8))
    assert set(detector.last_timings) == {"preprocess_ms", "inference_ms", "postprocess_ms"}
    assert all(v >= 0 for v in detector.last_timings.values())
    with pytest.raises(ValueError, match="always pads frames"):
        create_detector("onnx", onnx_file, letterbox="minimal")
