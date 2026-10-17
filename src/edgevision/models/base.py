"""Backend-independent object-detection types.

Every inference backend (PyTorch now, ONNX Runtime later) turns its own
library-specific output into the same :class:`Detection` records. Code that
uses detections (tracking, drawing, evaluation, comparing backends) therefore
never needs to know which library produced them.

Conventions shared by all backends:

* Input frames are NumPy ``uint8`` arrays of shape ``(height, width, 3)`` in
  OpenCV's **BGR** channel order, exactly as ``VideoReader`` and
  ``cv2.imread`` produce them.
* Boxes are ``(x1, y1, x2, y2)`` in pixels of the **original input frame**
  (not the resized model input): ``(x1, y1)`` is the top-left corner,
  ``(x2, y2)`` the bottom-right, with ``x1 <= x2`` and ``y1 <= y2``.
* ``confidence`` is the post-NMS score in ``[0, 1]``.

Importing this module does not require NumPy, PyTorch or Ultralytics.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

#: YOLO models downsample by 32, so the square input size must be a multiple of it.
MODEL_STRIDE = 32


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DetectorError(Exception):
    """Base class for detector problems."""


class DeviceUnavailableError(DetectorError):
    """The requested compute device (e.g. CUDA) cannot be used."""


class ModelLoadError(DetectorError):
    """The model file could not be found, downloaded, or loaded."""


class InferenceError(DetectorError):
    """Running the model on a frame failed."""


# ---------------------------------------------------------------------------
# Small validation helpers (bool is rejected everywhere: True is an int in Python)
# ---------------------------------------------------------------------------


def _is_integer(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _is_real(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _require_non_negative_int(name: str, value: Any) -> int:
    if not _is_integer(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return int(value)


def _require_fraction(name: str, value: Any) -> float:
    if not _is_real(value) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a number between 0 and 1, got {value!r}")
    return float(value)


# ---------------------------------------------------------------------------
# Detection record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Detection:
    """One detected object in one frame. Immutable and validated on creation."""

    frame_index: int
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str

    def __post_init__(self) -> None:
        # frozen=True blocks normal assignment, so normalised values are set
        # with object.__setattr__ (the documented way for frozen dataclasses).
        set_ = object.__setattr__
        set_(self, "frame_index", _require_non_negative_int("frame_index", self.frame_index))
        set_(self, "class_id", _require_non_negative_int("class_id", self.class_id))
        set_(self, "confidence", _require_fraction("confidence", self.confidence))

        if not isinstance(self.class_name, str) or not self.class_name.strip():
            raise ValueError(f"class_name must be a non-empty string, got {self.class_name!r}")

        box = self.bbox_xyxy
        if isinstance(box, (str, bytes)) or not hasattr(box, "__len__") or len(box) != 4:
            raise ValueError(f"bbox_xyxy must be four numbers (x1, y1, x2, y2), got {box!r}")
        if not all(_is_real(v) and math.isfinite(v) for v in box):
            raise ValueError(f"bbox_xyxy must contain only finite numbers, got {box!r}")
        x1, y1, x2, y2 = (float(v) for v in box)
        if x2 < x1 or y2 < y1:
            raise ValueError(
                f"bbox_xyxy must satisfy x1 <= x2 and y1 <= y2 (top-left, then bottom-right), "
                f"got {box!r}"
            )
        set_(self, "bbox_xyxy", (x1, y1, x2, y2))

    @property
    def width(self) -> float:
        return self.bbox_xyxy[2] - self.bbox_xyxy[0]

    @property
    def height(self) -> float:
        return self.bbox_xyxy[3] - self.bbox_xyxy[1]

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-friendly representation."""
        return {
            "frame_index": self.frame_index,
            "bbox_xyxy": list(self.bbox_xyxy),
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


# ---------------------------------------------------------------------------
# Settings shared by all backends
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectionSettings:
    """Inference settings that every backend must apply identically.

    The defaults match Ultralytics' own defaults and ``configs/default.yaml``
    (a test checks that they stay in sync with the YAML file).
    """

    input_size: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.7
    classes: tuple[int, ...] | None = None  # None = keep every class

    def __post_init__(self) -> None:
        size = self.input_size
        if not _is_integer(size) or size < MODEL_STRIDE or size % MODEL_STRIDE != 0:
            raise ValueError(
                f"input_size must be an integer >= {MODEL_STRIDE} and a multiple of "
                f"{MODEL_STRIDE}, got {size!r}"
            )
        object.__setattr__(self, "input_size", int(size))
        object.__setattr__(
            self, "conf_threshold", _require_fraction("conf_threshold", self.conf_threshold)
        )
        object.__setattr__(
            self, "iou_threshold", _require_fraction("iou_threshold", self.iou_threshold)
        )
        if self.classes is not None:
            classes = self.classes
            if isinstance(classes, (str, bytes)) or not hasattr(classes, "__iter__"):
                raise ValueError(f"classes must be None or a list of class IDs, got {classes!r}")
            classes = tuple(classes)
            if not classes or not all(_is_integer(c) and c >= 0 for c in classes):
                raise ValueError(
                    "classes must be None (all classes) or a non-empty list of non-negative "
                    f"integer class IDs, got {self.classes!r}"
                )
            object.__setattr__(self, "classes", tuple(int(c) for c in classes))

    @classmethod
    def from_model_config(cls, model_config: Any) -> DetectionSettings:
        """Build settings from the ``model`` section of an EdgeVision config."""
        return cls(
            input_size=model_config.input_size,
            conf_threshold=model_config.conf_threshold,
            iou_threshold=model_config.iou_threshold,
            classes=model_config.classes,
        )


# ---------------------------------------------------------------------------
# Input validation and the detector interface
# ---------------------------------------------------------------------------


def validate_bgr_frame(frame: Any) -> None:
    """Raise ``ValueError`` unless ``frame`` is a non-empty ``uint8`` BGR image array."""
    try:
        import numpy as np
    except ImportError:  # without NumPy nothing can be a valid frame
        raise ValueError("frame must be a NumPy array, but NumPy is not installed") from None

    if not isinstance(frame, np.ndarray):
        raise ValueError(f"frame must be a NumPy array, got {type(frame).__name__}")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            f"frame must have shape (height, width, 3) with BGR channels, got shape {frame.shape}"
        )
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError(f"frame is empty (shape {frame.shape})")
    if frame.dtype != np.uint8:
        raise ValueError(f"frame must have dtype uint8 (0-255 pixel values), got {frame.dtype}")


def validate_frame_index(frame_index: Any) -> int:
    return _require_non_negative_int("frame_index", frame_index)


@runtime_checkable
class Detector(Protocol):
    """What every detection backend provides.

    ``detect`` takes one BGR frame (see module docstring) and returns the
    objects found in it, with boxes in that frame's pixel coordinates and
    ``frame_index`` copied into every record. It returns an empty list when
    nothing is detected.
    """

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> list[Detection]: ...

    def close(self) -> None: ...
