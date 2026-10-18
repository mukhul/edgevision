"""Object detection with a pretrained Ultralytics YOLO model running on PyTorch.

Usage::

    from edgevision.models import DetectionSettings, PytorchYoloDetector

    with PytorchYoloDetector("models/yolov8n.pt", device="auto") as detector:
        detections = detector.detect(frame, frame_index=0)  # frame: BGR uint8 array

What this class does, and what it leaves to Ultralytics:

* **EdgeVision** validates inputs and settings, chooses the device, and turns
  Ultralytics' output into backend-independent :class:`Detection` records.
* **Ultralytics** does the actual computer vision: preprocessing (letterbox
  resize to ``input_size``, BGR->RGB, scaling to 0-1), the forward pass,
  non-maximum suppression, and scaling boxes back to the original frame.

PyTorch and Ultralytics are optional dependencies (the ``detector`` extra)
and are imported only when a detector is created. Official model names such
as ``yolov8n.pt`` are downloaded by Ultralytics the first time they are loaded
(to the given path, e.g. ``models/yolov8n.pt``); nothing is downloaded at
install time or by importing this module.
"""

from __future__ import annotations

import logging
import os
from types import TracebackType
from typing import TYPE_CHECKING, Any

from edgevision.models.base import (
    Detection,
    DetectionSettings,
    DetectorError,
    DeviceUnavailableError,
    InferenceError,
    ModelLoadError,
    validate_bgr_frame,
    validate_frame_index,
)
from edgevision.optional import import_optional

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

DEVICES = ("auto", "cpu", "cuda")
LETTERBOX_MODES = ("minimal", "square")


def _stage_timings(speed: Any) -> dict[str, float] | None:
    """Ultralytics' ``Results.speed`` ({'preprocess': ms, ...}) as EdgeVision stage timings."""
    if not isinstance(speed, dict):
        return None
    try:
        return {f"{stage}_ms": float(speed[stage]) for stage in ("preprocess", "inference", "postprocess")}
    except (KeyError, TypeError, ValueError):
        return None


def _load_torch() -> Any:
    return import_optional(
        "torch", display_name="PyTorch", extra="detector", purpose="run the PyTorch detector"
    )


def _load_ultralytics() -> Any:
    return import_optional(
        "ultralytics", display_name="Ultralytics", extra="detector", purpose="load YOLO models"
    )


def resolve_device(requested: Any, torch: Any) -> str:
    """Turn ``"auto"`` / ``"cpu"`` / ``"cuda"`` into the device PyTorch should use.

    ``auto`` picks CUDA only if PyTorch itself reports it usable, else CPU.
    An explicit ``cuda`` request is never silently downgraded to CPU.
    """
    if not isinstance(requested, str) or requested not in DEVICES:
        raise ValueError(f"device must be one of {list(DEVICES)}, got {requested!r}")
    if requested == "cpu":
        return "cpu"
    cuda_available = bool(torch.cuda.is_available())
    if requested == "auto":
        return "cuda:0" if cuda_available else "cpu"
    if not cuda_available:
        build = torch.version.cuda or "none (CPU-only build)"
        raise DeviceUnavailableError(
            "device='cuda' was requested, but PyTorch reports that CUDA is not available "
            f"(PyTorch {torch.__version__}, CUDA build: {build}). Use device='cpu' or "
            "device='auto', or run on a machine with an NVIDIA GPU and a CUDA-enabled PyTorch build."
        )
    return "cuda:0"


class PytorchYoloDetector:
    """Ultralytics YOLO detector implementing the :class:`~edgevision.models.base.Detector` protocol.

    Args:
        model: Path to a ``.pt`` detection model, or an official name such as
            ``"yolov8n.pt"`` (downloaded on first use).
        device: ``"auto"`` (default), ``"cpu"`` or ``"cuda"``.
        settings: Input size, thresholds and class filter. Defaults to
            :class:`DetectionSettings` defaults.
        letterbox: How Ultralytics pads a frame to the model input.
            ``"minimal"`` (default, Ultralytics' normal behaviour) pads only to
            the next multiple of 32, e.g. 1024x576 -> 640x384. ``"square"`` pads
            to the full ``input_size`` square, exactly like the ONNX backend,
            so both backends process identical input tensors (used by benchmarks).

    Raises:
        MissingDependencyError: PyTorch or Ultralytics is not installed.
        DeviceUnavailableError: ``device="cuda"`` but CUDA is not usable.
        ModelLoadError: the model cannot be loaded or is not a detection model.
    """

    backend_name = "pytorch"
    description = "PyTorch via Ultralytics"  # short runtime description, for summaries

    def __init__(
        self,
        model: str | os.PathLike[str],
        *,
        device: str = "auto",
        settings: DetectionSettings | None = None,
        letterbox: str = "minimal",
    ) -> None:
        if letterbox not in LETTERBOX_MODES:
            raise ValueError(f"letterbox must be one of {list(LETTERBOX_MODES)}, got {letterbox!r}")
        if settings is None:
            settings = DetectionSettings()
        if not isinstance(settings, DetectionSettings):
            raise TypeError(f"settings must be a DetectionSettings, got {type(settings).__name__}")
        if not isinstance(model, (str, os.PathLike)) or not str(model).strip():
            raise ValueError(f"model must be a non-empty path or model name, got {model!r}")

        self.settings = settings
        self.model_path = os.fspath(model)
        self.requested_device = device
        self.letterbox = letterbox
        #: Stage times of the most recent detect() call in milliseconds, as measured by
        #: Ultralytics itself (``Results.speed``, time.perf_counter() with GPU synchronisation):
        #: preprocess (letterbox + tensor), inference (forward pass) and postprocess (NMS,
        #: scaling boxes, building Ultralytics results). None if Ultralytics reports none.
        self.last_timings: dict[str, float] | None = None

        torch = _load_torch()
        self.device = resolve_device(device, torch)  # before loading: fail fast on bad devices
        self._torch = torch
        self._model: Any = self._load_model(_load_ultralytics())
        self.class_names: dict[int, str] = {
            int(class_id): str(name) for class_id, name in self._model.names.items()
        }
        logger.info(
            "Loaded %s on %s (%d classes)", self.model_path, self.device, len(self.class_names)
        )

    def _load_model(self, ultralytics: Any) -> Any:
        try:
            model = ultralytics.YOLO(self.model_path)
        except Exception as exc:
            raise ModelLoadError(
                f"Could not load YOLO model {self.model_path!r}: {type(exc).__name__}: {exc}. "
                "Official names such as 'yolov8n.pt' are downloaded on first use (this needs an "
                "internet connection); any other path must point to an existing .pt file."
            ) from exc
        task = getattr(model, "task", None)
        if task != "detect":
            raise ModelLoadError(
                f"{self.model_path!r} is a {task!r} model, but EdgeVision needs an object "
                "detection ('detect') model such as yolov8n.pt."
            )
        return model

    # --- inference -----------------------------------------------------------

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> list[Detection]:
        """Detect objects in one BGR frame. Returns an empty list if nothing is found."""
        frame_index = validate_frame_index(frame_index)
        validate_bgr_frame(frame)
        model = self._require_open()
        s = self.settings
        try:
            results = model.predict(
                source=frame,
                imgsz=s.input_size,
                conf=s.conf_threshold,
                iou=s.iou_threshold,
                classes=list(s.classes) if s.classes is not None else None,
                device=self.device,
                verbose=False,  # no per-frame log lines
                **({"rect": False} if self.letterbox == "square" else {}),
            )
        except Exception as exc:
            raise InferenceError(
                f"YOLO inference failed on frame {frame_index} (shape {frame.shape}, "
                f"device {self.device}): {type(exc).__name__}: {exc}"
            ) from exc
        if len(results) != 1:
            raise InferenceError(f"Expected 1 result for 1 frame, got {len(results)}")
        self.last_timings = _stage_timings(getattr(results[0], "speed", None))
        return self._to_detections(results[0], frame_index)

    def _to_detections(self, result: Any, frame_index: int) -> list[Detection]:
        """Convert one Ultralytics ``Results`` object into Detection records."""
        boxes = result.boxes
        if boxes is None:
            return []
        # .tolist() copies tensors to plain Python numbers (from GPU memory if needed).
        rows = boxes.xyxy.tolist()  # boxes already scaled back to the original frame
        scores = boxes.conf.tolist()
        class_ids = boxes.cls.tolist()
        if not len(rows) == len(scores) == len(class_ids):
            raise InferenceError(
                f"Inconsistent model output: {len(rows)} boxes, {len(scores)} scores, "
                f"{len(class_ids)} class IDs"
            )

        detections = []
        for xyxy, score, raw_class in zip(rows, scores, class_ids):
            class_id = int(round(raw_class))
            name = self.class_names.get(class_id)
            if name is None:
                raise InferenceError(f"Model returned class ID {class_id}, which has no class name")
            try:
                detections.append(Detection(frame_index, tuple(xyxy), score, class_id, name))
            except ValueError as exc:
                raise InferenceError(f"Model produced an invalid detection: {exc}") from exc
        return detections

    # --- resource handling ---------------------------------------------------

    def _require_open(self) -> Any:
        if self._model is None:
            raise DetectorError("This detector has been closed; create a new one.")
        return self._model

    @property
    def closed(self) -> bool:
        return self._model is None

    def close(self) -> None:
        """Drop the model so its memory can be freed. Safe to call more than once."""
        if self._model is None:
            return
        self._model = None
        if self.device.startswith("cuda"):
            self._torch.cuda.empty_cache()  # return cached GPU memory to the driver

    def __enter__(self) -> PytorchYoloDetector:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self.closed else "open"
        return f"PytorchYoloDetector({self.model_path!r}, device={self.device!r}, {state})"
