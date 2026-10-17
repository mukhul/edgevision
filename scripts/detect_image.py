"""Run the PyTorch YOLO detector on one image and print what it finds.

Run from the repository root:

    python scripts/detect_image.py path/to/photo.jpg
    python scripts/detect_image.py path/to/photo.jpg --device cpu --conf 0.4 --json results/photo.json

Defaults for the model, input size, thresholds and class filter come from
configs/default.yaml; command-line options override them. The first run
downloads yolov8n.pt to models/ (about 6 MB).

Exit status: 0 success, 1 the image/model/inference failed, 2 invalid arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import edgevision
from edgevision.config import DEFAULT_CONFIG_PATH, ConfigError, load_config
from edgevision.models import Detection, DetectionSettings, DetectorError, PytorchYoloDetector
from edgevision.optional import MissingDependencyError, import_optional

EXAMPLES = """\
examples:
  python scripts/detect_image.py data/input/photo.jpg
  python scripts/detect_image.py data/input/photo.jpg --device cpu --conf 0.4
  python scripts/detect_image.py data/input/photo.jpg --json results/photo_detections.json
"""


class ImageError(Exception):
    """The input image is missing or cannot be decoded."""


# --- argument parsing --------------------------------------------------------


def fraction(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a number between 0 and 1, got {text!r}") from None
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1, got {value}")
    return value


def positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/detect_image.py",
        description="Detect objects in one image with the PyTorch YOLO backend.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", type=Path, help="image file (JPEG, PNG, BMP, ...)")
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
    parser.add_argument("--json", type=Path, metavar="PATH", help="also write the results to this JSON file")
    parser.add_argument("--overwrite", action="store_true", help="allow --json to replace an existing file")
    return parser


# --- steps -------------------------------------------------------------------


def read_image(path: Path) -> Any:
    """Read an image as a BGR uint8 array (works with non-ASCII paths on Windows too)."""
    if not path.exists():
        raise ImageError(f"Image not found: {path} (looked relative to {Path.cwd()})")
    if not path.is_file():
        raise ImageError(f"Image path is not a file: {path}")
    cv2 = import_optional("cv2", display_name="OpenCV", extra="vision", purpose="read images")
    import numpy as np

    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise ImageError(f"Could not decode {path} as an image (is it a JPEG/PNG/BMP file?)")
    return image


def format_summary(
    image_path: Path,
    image_size: tuple[int, int],
    model: str,
    device: str,
    settings: DetectionSettings,
    detections: list[Detection],
) -> str:
    classes = "all" if settings.classes is None else ", ".join(map(str, settings.classes))
    lines = [
        f"Image        {image_path} ({image_size[0]} x {image_size[1]} px)",
        f"Model        {model} (PyTorch via Ultralytics, device: {device})",
        f"Settings     input size {settings.input_size}, confidence >= {settings.conf_threshold}, "
        f"NMS IoU {settings.iou_threshold}, classes: {classes}",
        f"Detections   {len(detections)}",
    ]
    if detections:
        lines += ["", f"  {'#':>3}  {'class':<16}{'conf':>6}  {'x1':>7} {'y1':>7} {'x2':>7} {'y2':>7}  (pixels)"]
        ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
        for i, d in enumerate(ordered, start=1):
            x1, y1, x2, y2 = d.bbox_xyxy
            lines.append(
                f"  {i:>3}  {d.class_name:<16}{d.confidence:>6.3f}  {x1:>7.1f} {y1:>7.1f} {x2:>7.1f} {y2:>7.1f}"
            )
    return "\n".join(lines)


def results_document(
    image_path: Path,
    image_size: tuple[int, int],
    model: str,
    device: str,
    settings: DetectionSettings,
    detections: list[Detection],
) -> dict[str, Any]:
    return {
        "edgevision_version": edgevision.__version__,
        "backend": "pytorch",
        "image": image_path.as_posix(),
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "model": model,
        "device": device,
        "settings": {
            "input_size": settings.input_size,
            "conf_threshold": settings.conf_threshold,
            "iou_threshold": settings.iou_threshold,
            "classes": None if settings.classes is None else list(settings.classes),
        },
        "bbox_format": "xyxy, original image pixels",
        "detections": [d.to_dict() for d in detections],
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        model_config = load_config(args.config).model
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    try:
        settings = DetectionSettings(
            input_size=args.imgsz if args.imgsz is not None else model_config.input_size,
            conf_threshold=args.conf if args.conf is not None else model_config.conf_threshold,
            iou_threshold=args.iou if args.iou is not None else model_config.iou_threshold,
            classes=model_config.classes,
        )
    except ValueError as exc:
        parser.error(str(exc))  # exits with status 2
    model = args.model or model_config.weights.as_posix()

    if args.json is not None and args.json.exists() and not args.overwrite:
        print(f"ERROR: {args.json} already exists; pass --overwrite to replace it.", file=sys.stderr)
        return 1

    try:
        image = read_image(args.image)
        if not Path(model).exists():
            print(f"Note: {model} not found locally; Ultralytics will try to download it.", file=sys.stderr)
        with PytorchYoloDetector(model, device=args.device, settings=settings) as detector:
            device = detector.device
            detections = detector.detect(image, frame_index=0)
    except (ImageError, MissingDependencyError, DetectorError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    image_size = (image.shape[1], image.shape[0])
    print(format_summary(args.image, image_size, model, device, settings, detections))

    if args.json is not None:
        document = results_document(args.image, image_size, model, device, settings, detections)
        try:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(document, indent=2), encoding="utf-8")
        except OSError as exc:
            print(f"ERROR: could not write {args.json}: {exc}", file=sys.stderr)
            return 1
        print(f"\nSaved JSON: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
