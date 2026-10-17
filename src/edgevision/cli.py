"""Command-line helpers shared by the detection scripts.

Both ``scripts/detect_image.py`` and ``scripts/detect_video.py`` take their
model, input size, thresholds and class filter from the config file and let
command-line options override them. Keeping that logic here means the two
commands can never drift apart in their defaults.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from edgevision.config import DEFAULT_CONFIG_PATH, ModelConfig
from edgevision.models.base import DetectionSettings


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


def add_detector_arguments(parser: argparse.ArgumentParser) -> None:
    """Add --model, --device, --conf, --iou, --imgsz and --config."""
    parser.add_argument("--model", help="model name or .pt path (default: model.weights from the config)")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto",
        help="compute device (default: auto = CUDA if PyTorch can use it, else CPU)",
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
    return args.model or model_config.weights.as_posix()


def describe_settings(settings: DetectionSettings) -> str:
    classes = "all" if settings.classes is None else ", ".join(map(str, settings.classes))
    return (
        f"input size {settings.input_size}, confidence >= {settings.conf_threshold}, "
        f"NMS IoU {settings.iou_threshold}, classes: {classes}"
    )

