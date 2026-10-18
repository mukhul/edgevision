"""Object-detection models.

Importing this package does not require PyTorch, Ultralytics, ONNX Runtime or
OpenCV; each backend imports its libraries only when a detector is created.
"""

from edgevision.models.base import (
    Detection,
    DetectionSettings,
    Detector,
    DetectorError,
    DeviceUnavailableError,
    InferenceError,
    ModelLoadError,
)
from edgevision.models.factory import BACKENDS, DEFAULT_BACKEND, create_detector
from edgevision.models.onnx_backend import OnnxYoloDetector
from edgevision.models.pytorch_backend import PytorchYoloDetector

__all__ = [
    "BACKENDS",
    "DEFAULT_BACKEND",
    "Detection",
    "DetectionSettings",
    "Detector",
    "DetectorError",
    "DeviceUnavailableError",
    "InferenceError",
    "ModelLoadError",
    "OnnxYoloDetector",
    "PytorchYoloDetector",
    "create_detector",
]
