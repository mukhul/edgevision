"""Command-line helpers shared by the detection scripts.

Both ``scripts/detect_image.py`` and ``scripts/detect_video.py`` take their
model, input size, thresholds and class filter from the config file and let
command-line options override them. Keeping that logic here means the two
commands can never drift apart in their defaults.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from edgevision.config import DEFAULT_CONFIG_PATH, ModelConfig
from edgevision.models.base import DetectionSettings
from edgevision.models.factory import BACKENDS, DEFAULT_BACKEND


def fraction(text: str) -> float:
    """argparse type: a number between 0 and 1."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a number between 0 and 1, got {text!r}") from None
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1, got {value}")
    return value


def positive_int(text: str) -> int:
    """argparse type: an integer >= 1."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return value


def positive_float(text: str) -> float:
    """argparse type: a finite number > 0 (e.g. a frame rate)."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a positive number, got {text!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"must be a finite number greater than 0, got {text!r}")
    return value


def add_detector_arguments(parser: argparse.ArgumentParser) -> None:
    """Add --backend, --model, --device, --conf, --iou, --imgsz and --config."""
    parser.add_argument(
        "--backend", choices=list(BACKENDS), default=DEFAULT_BACKEND,
        help="inference backend: pytorch (Ultralytics) or onnx (ONNX Runtime with an exported model) "
        f"(default: {DEFAULT_BACKEND})",
    )
    parser.add_argument(
        "--model",
        help="model file for the chosen backend (default: model.weights from the config for pytorch, "
        "model.onnx_path for onnx)",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto",
        help="compute device (default: auto = CUDA if the backend can use it, else CPU)",
    )
    parser.add_argument("--conf", type=fraction, help="confidence threshold 0-1 (default: from config)")
    parser.add_argument("--iou", type=fraction, help="NMS IoU threshold 0-1 (default: from config)")
    parser.add_argument(
        "--imgsz", type=positive_int, help="model input size, multiple of 32 (default: from config)"
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help=f"config file supplying defaults (default: {DEFAULT_CONFIG_PATH.as_posix()})",
    )


def detection_settings(args: argparse.Namespace, model_config: ModelConfig) -> DetectionSettings:
    """Config values overridden by any options given. Raises ValueError for invalid combinations."""
    return DetectionSettings(
        input_size=args.imgsz if args.imgsz is not None else model_config.input_size,
        conf_threshold=args.conf if args.conf is not None else model_config.conf_threshold,
        iou_threshold=args.iou if args.iou is not None else model_config.iou_threshold,
        classes=model_config.classes,
    )


def model_name(args: argparse.Namespace, model_config: ModelConfig) -> str:
    """The model to load: --model, else the config's path for the selected backend."""
    if args.model:
        return args.model
    if getattr(args, "backend", DEFAULT_BACKEND) == "onnx":
        return model_config.onnx_path.as_posix()
    return model_config.weights.as_posix()


def detector_details(detector: object) -> dict[str, object]:
    """Extra, backend-specific facts to record in JSON results (empty for PyTorch).

    For ONNX Runtime, this is the list of execution providers the session
    actually uses, so a result file always shows whether a GPU was involved.
    """
    providers = getattr(detector, "providers", None)
    return {} if providers is None else {"execution_providers": list(providers)}


def describe_settings(settings: DetectionSettings) -> str:
    classes = "all" if settings.classes is None else ", ".join(map(str, settings.classes))
    return (
        f"input size {settings.input_size}, confidence >= {settings.conf_threshold}, "
        f"NMS IoU {settings.iou_threshold}, classes: {classes}"
    )

