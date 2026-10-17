"""Object-detection models.

Importing this package does not require PyTorch or Ultralytics; the PyTorch
backend imports them only when a detector is created.
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
from edgevision.models.pytorch_backend import PytorchYoloDetector

__all__ = [
    "Detection",
    "DetectionSettings",
    "Detector",
    "DetectorError",
    "DeviceUnavailableError",
    "InferenceError",
    "ModelLoadError",
    "PytorchYoloDetector",
]
