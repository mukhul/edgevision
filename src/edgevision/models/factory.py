"""Create a detector for a named backend.

The commands use this so that choosing ``--backend pytorch`` or ``--backend
onnx`` is the only difference between the two code paths: both return an
object with the same :class:`~edgevision.models.base.Detector` interface, and
everything after detection (tracking, drawing, JSON, video) is shared.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from edgevision.models.base import DetectionSettings

BACKENDS = ("pytorch", "onnx")
DEFAULT_BACKEND = "pytorch"


def check_model_matches_backend(backend: str, model: str | os.PathLike[str]) -> None:
    """Raise ValueError if the model file obviously belongs to the other backend.

    Ultralytics would quietly load an ``.onnx`` file given to the PyTorch
    backend and run it through ONNX Runtime, while every summary and JSON file
    said "pytorch". That mislabelling is refused here.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {list(BACKENDS)}, got {backend!r}")
    suffix = Path(os.fspath(model)).suffix.lower()
    if backend == "pytorch" and suffix == ".onnx":
        raise ValueError(f"{model} is an ONNX model; use --backend onnx to run it with ONNX Runtime")
    if backend == "onnx" and suffix != ".onnx":
        raise ValueError(
            f"--backend onnx needs an .onnx file, got {model}. Export one with: python scripts/export_onnx.py"
        )


def create_detector(
    backend: str,
    model: str | os.PathLike[str],
    *,
    device: str = "auto",
    settings: DetectionSettings | None = None,
    letterbox: str | None = None,
) -> Any:
    """Return a ``PytorchYoloDetector`` or ``OnnxYoloDetector`` (both implement ``Detector``).

    ``letterbox`` is ``None`` for each backend's normal behaviour, ``"minimal"``
    (PyTorch only) or ``"square"`` (ONNX always uses the full square input).
    """
    check_model_matches_backend(backend, model)
    if backend == "onnx":
        if letterbox not in (None, "square"):
            raise ValueError("the ONNX backend always pads frames to the model's square input (letterbox 'square')")
        from edgevision.models.onnx_backend import OnnxYoloDetector

        return OnnxYoloDetector(model, device=device, settings=settings)
    from edgevision.models.pytorch_backend import PytorchYoloDetector

    return PytorchYoloDetector(model, device=device, settings=settings, letterbox=letterbox or "minimal")
