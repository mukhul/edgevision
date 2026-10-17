"""Tests for edgevision.models.pytorch_backend.

Unit tests use the ``fake_yolo`` fixture (tests/conftest.py): PyTorch and
Ultralytics are replaced by fakes, so nothing is downloaded and no real model
runs. They check *our* code: validation, device choice, the arguments passed
to Ultralytics, conversion to Detection records, and error handling.

The real-model test at the end is opt-in (see its skip reason).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from edgevision.models import (
    Detection,
    DetectionSettings,
    Detector,
    DetectorError,
    DeviceUnavailableError,
    InferenceError,
    ModelLoadError,
    PytorchYoloDetector,
)
from edgevision.models.pytorch_backend import resolve_device
from edgevision.optional import MissingDependencyError

REPO = Path(__file__).resolve().parents[1]
FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def make_detector(**kwargs: Any) -> PytorchYoloDetector:
    return PytorchYoloDetector("models/yolov8n.pt", **kwargs)


# --- interface and arguments passed to Ultralytics -----------------------------


def test_detector_satisfies_the_detector_protocol(fake_yolo) -> None:
    assert isinstance(make_detector(), Detector)


def test_model_path_is_passed_to_ultralytics(fake_yolo) -> None:
    PytorchYoloDetector(Path("models") / "custom.pt")
    assert fake_yolo.loaded_paths == [os.path.join("models", "custom.pt")]


def test_default_settings_are_passed_to_predict(fake_yolo) -> None:
    make_detector(device="cpu").detect(FRAME)
    call = fake_yolo.predict_calls[0]
    assert call["source"] is FRAME  # the original BGR frame, untouched
    assert call == {
        "source": FRAME,
        "imgsz": 640,
        "conf": 0.25,
        "iou": 0.7,
        "classes": None,
        "device": "cpu",
        "verbose": False,
    }


def test_custom_settings_are_passed_to_predict(fake_yolo) -> None:
    settings = DetectionSettings(input_size=320, conf_threshold=0.5, iou_threshold=0.45, classes=[15])
    make_detector(settings=settings).detect(FRAME)
    call = fake_yolo.predict_calls[0]
    assert (call["imgsz"], call["conf"], call["iou"], call["classes"]) == (320, 0.5, 0.45, [15])


# --- converting model output to Detection records ------------------------------


def test_model_output_is_converted_to_detections(fake_yolo) -> None:
    fake_yolo.boxes = [
        (10.0, 20.0, 110.0, 220.0, 0.91, 15),
        (0.0, 5.5, 639.0, 479.0, 0.42, 5),
    ]
    detections = make_detector().detect(FRAME, frame_index=42)

    assert detections == [
        Detection(42, (10.0, 20.0, 110.0, 220.0), 0.91, 15, "cat"),
        Detection(42, (0.0, 5.5, 639.0, 479.0), 0.42, 5, "bus"),
    ]


def test_frame_index_is_propagated_to_every_detection(fake_yolo) -> None:
    fake_yolo.boxes = [(1, 1, 2, 2, 0.5, 0)] * 3
    detector = make_detector()
    for index in (0, 7, 12345):
        assert {d.frame_index for d in detector.detect(FRAME, frame_index=index)} == {index}


def test_no_objects_gives_an_empty_list(fake_yolo) -> None:
    assert make_detector().detect(FRAME) == []


def test_missing_boxes_gives_an_empty_list(fake_yolo) -> None:
    fake_yolo.boxes_is_none = True
    assert make_detector().detect(FRAME) == []


def test_unknown_class_id_is_an_inference_error(fake_yolo) -> None:
    fake_yolo.boxes = [(1, 1, 2, 2, 0.5, 99)]
    with pytest.raises(InferenceError, match="class ID 99"):
        make_detector().detect(FRAME)


def test_invalid_box_from_model_is_an_inference_error(fake_yolo) -> None:
    fake_yolo.boxes = [(50, 1, 10, 2, 0.5, 0)]  # x1 > x2
    with pytest.raises(InferenceError, match="invalid detection"):
        make_detector().detect(FRAME)


def test_wrong_number_of_results_is_an_inference_error(fake_yolo) -> None:
    fake_yolo.results_per_call = 2
    with pytest.raises(InferenceError, match="Expected 1 result"):
        make_detector().detect(FRAME)


# --- invalid inputs --------------------------------------------------------------


@pytest.mark.parametrize(
    "frame",
    [
        None,
        np.zeros((0, 0, 3), dtype=np.uint8),
        np.zeros((480, 640), dtype=np.uint8),
        np.zeros((480, 640, 4), dtype=np.uint8),
        np.zeros((480, 640, 3), dtype=np.float32),
        [[0, 0, 0]],
    ],
)
def test_invalid_frame_is_rejected_before_inference(fake_yolo, frame: Any) -> None:
    detector = make_detector()
    with pytest.raises(ValueError):
        detector.detect(frame)
    assert fake_yolo.predict_calls == []


@pytest.mark.parametrize("frame_index", [-1, True, 1.5, "0"])
def test_invalid_frame_index_is_rejected(fake_yolo, frame_index: Any) -> None:
    with pytest.raises(ValueError, match="frame_index"):
        make_detector().detect(FRAME, frame_index=frame_index)


@pytest.mark.parametrize("model", ["", "   ", 123, None])
def test_invalid_model_argument(fake_yolo, model: Any) -> None:
    with pytest.raises(ValueError, match="model must be"):
        PytorchYoloDetector(model)


def test_settings_must_be_detection_settings(fake_yolo) -> None:
    with pytest.raises(TypeError, match="DetectionSettings"):
        make_detector(settings={"conf_threshold": 0.5})


# --- device selection ------------------------------------------------------------


@pytest.mark.parametrize(
    ("requested", "cuda_available", "expected"),
    [
        ("auto", False, "cpu"),
        ("auto", True, "cuda:0"),
        ("cpu", False, "cpu"),
        ("cpu", True, "cpu"),  # an explicit CPU request is respected even with a GPU
        ("cuda", True, "cuda:0"),
    ],
)
def test_device_selection(fake_yolo, requested: str, cuda_available: bool, expected: str) -> None:
    fake_yolo.cuda_available = cuda_available
    detector = make_detector(device=requested)
    assert detector.device == expected
    detector.detect(FRAME)
    assert fake_yolo.predict_calls[0]["device"] == expected


def test_explicit_cuda_without_cuda_fails_instead_of_using_cpu(fake_yolo) -> None:
    fake_yolo.cuda_available = False
    with pytest.raises(DeviceUnavailableError) as info:
        make_detector(device="cuda")
    message = str(info.value)
    assert "CUDA is not available" in message
    assert "CPU-only build" in message
    assert fake_yolo.loaded_paths == []  # fails before spending time loading the model


@pytest.mark.parametrize("device", ["gpu", "CUDA", "cuda:1", "", None, 0])
def test_unknown_device_is_rejected(fake_yolo, device: Any) -> None:
    with pytest.raises(ValueError, match="device must be one of"):
        make_detector(device=device)


def test_resolve_device_does_not_query_cuda_for_cpu() -> None:
    class NoCudaTorch:
        @property
        def cuda(self) -> None:
            raise AssertionError("CUDA should not be queried for device='cpu'")

    assert resolve_device("cpu", NoCudaTorch()) == "cpu"


# --- loading and inference failures ------------------------------------------------


def test_model_loading_failure_is_reported(fake_yolo) -> None:
    fake_yolo.load_error = FileNotFoundError("models/missing.pt does not exist")
    with pytest.raises(ModelLoadError) as info:
        PytorchYoloDetector("models/missing.pt")
    message = str(info.value)
    assert "models/missing.pt" in message
    assert "FileNotFoundError" in message
    assert isinstance(info.value.__cause__, FileNotFoundError)  # original error kept


def test_non_detection_model_is_rejected(fake_yolo) -> None:
    fake_yolo.task = "segment"
    with pytest.raises(ModelLoadError, match="'segment' model"):
        make_detector()


def test_inference_failure_is_reported(fake_yolo) -> None:
    fake_yolo.predict_error = RuntimeError("out of memory")
    with pytest.raises(InferenceError) as info:
        make_detector().detect(FRAME, frame_index=9)
    message = str(info.value)
    assert "frame 9" in message and "out of memory" in message
    assert isinstance(info.value.__cause__, RuntimeError)


# --- missing dependencies ----------------------------------------------------------


@pytest.mark.parametrize(("module", "name"), [("torch", "PyTorch"), ("ultralytics", "Ultralytics")])
def test_missing_dependency_gives_install_instructions(
    fake_yolo, monkeypatch: pytest.MonkeyPatch, module: str, name: str
) -> None:
    monkeypatch.setitem(sys.modules, module, None)  # behaves like "not installed"
    with pytest.raises(MissingDependencyError) as info:
        make_detector()
    message = str(info.value)
    assert f"{name} is required" in message
    assert 'pip install -e ".[detector]"' in message


def test_base_package_imports_without_ml_dependencies() -> None:
    code = (
        "import sys\n"
        "for name in ('torch', 'ultralytics', 'cv2', 'numpy'):\n"
        "    sys.modules[name] = None  # make each one unimportable\n"
        "import edgevision, edgevision.config, edgevision.device, edgevision.video\n"
        "from edgevision.models import Detection, DetectionSettings, PytorchYoloDetector\n"
        "d = Detection(0, (1, 2, 3, 4), 0.5, 0, 'person')\n"
        "print('ok', d.class_name, DetectionSettings().input_size)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok person 640"


# --- closing -----------------------------------------------------------------------


def test_close_releases_the_model(fake_yolo) -> None:
    detector = make_detector()
    detector.close()
    detector.close()  # safe to repeat
    assert detector.closed
    with pytest.raises(DetectorError, match="closed"):
        detector.detect(FRAME)


def test_closing_a_cuda_detector_frees_cached_gpu_memory(fake_yolo) -> None:
    fake_yolo.cuda_available = True
    detector = make_detector(device="cuda")
    detector.close()
    detector.close()
    assert fake_yolo.empty_cache_calls == 1


def test_context_manager_closes_even_on_error(fake_yolo) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with make_detector() as detector:
            raise RuntimeError("boom")
    assert detector.closed


# --- real model (opt-in) -------------------------------------------------------------

RUN_MODEL_TESTS = os.environ.get("EDGEVISION_RUN_MODEL_TESTS") == "1"
WEIGHTS = Path(os.environ.get("EDGEVISION_MODEL_WEIGHTS", REPO / "models" / "yolov8n.pt"))


@pytest.mark.model
@pytest.mark.skipif(
    not RUN_MODEL_TESTS,
    reason="real-model test is opt-in: set EDGEVISION_RUN_MODEL_TESTS=1 (needs the detector extra and models/yolov8n.pt)",
)
def test_real_yolov8n_inference_on_cpu() -> None:
    """Runs the genuine pretrained model on a photo that ships with Ultralytics."""
    if not WEIGHTS.is_file():
        pytest.skip(
            f"weights not found at {WEIGHTS}; download them once by running "
            "scripts/detect_image.py (tests never download)"
        )
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("ultralytics")
    from ultralytics.utils import ASSETS  # bus.jpg is part of the installed package

    image = cv2.imread(str(ASSETS / "bus.jpg"))
    assert image is not None
    height, width = image.shape[:2]

    with PytorchYoloDetector(WEIGHTS, device="cpu") as detector:
        first = detector.detect(image, frame_index=3)
        second = detector.detect(image, frame_index=3)

    names = [d.class_name for d in first]
    assert "bus" in names
    assert names.count("person") >= 3
    assert max(d.confidence for d in first if d.class_name == "bus") > 0.5
    for d in first:
        x1, y1, x2, y2 = d.bbox_xyxy
        assert d.frame_index == 3
        assert 0 <= x1 <= x2 <= width and 0 <= y1 <= y2 <= height  # original-image pixels
    # Same input, same CPU model -> same output.
    assert [(d.class_id, round(d.confidence, 4)) for d in first] == [
        (d.class_id, round(d.confidence, 4)) for d in second
    ]
