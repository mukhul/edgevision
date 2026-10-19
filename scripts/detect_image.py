"""Run a YOLO detector on one image and print what it finds.

Run from the repository root:

    python scripts/detect_image.py path/to/photo.jpg
    python scripts/detect_image.py path/to/photo.jpg --device cpu --conf 0.4 --json results/photo.json
    python scripts/detect_image.py path/to/photo.jpg --backend onnx

--backend pytorch (the default) runs the .pt model with Ultralytics;
--backend onnx runs an exported .onnx model with ONNX Runtime. Defaults for
the model, input size, thresholds and class filter come from
configs/default.yaml; command-line options override them. With the PyTorch
backend, the first run downloads yolov8n.pt to models/ (about 6 MB); ONNX
models are never downloaded.

Exit status: 0 success, 1 the image/model/inference failed, 2 invalid arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import edgevision
from edgevision.cli import (
    add_detector_arguments,
    describe_settings,
    detection_settings,
    detector_details,
    model_name,
    pytorch_letterbox,
)
from edgevision.config import ConfigError, load_config
from edgevision.models import Detection, DetectionSettings, DetectorError, create_detector
from edgevision.models.factory import check_model_matches_backend
from edgevision.optional import MissingDependencyError, import_optional

EXAMPLES = """\
examples:
  python scripts/detect_image.py data/input/photo.jpg
  python scripts/detect_image.py data/input/photo.jpg --device cpu --conf 0.4
  python scripts/detect_image.py data/input/photo.jpg --json results/photo_detections.json
  python scripts/detect_image.py data/input/photo.jpg --backend onnx --device cpu
"""


class ImageError(Exception):
    """The input image is missing or cannot be decoded."""


# --- argument parsing --------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/detect_image.py",
        description="Detect objects in one image with a YOLO model on PyTorch or ONNX Runtime.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", type=Path, help="image file (JPEG, PNG, BMP, ...)")
    add_detector_arguments(parser)
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
    runtime: str = "PyTorch via Ultralytics",
) -> str:
    lines = [
        f"Image        {image_path} ({image_size[0]} x {image_size[1]} px)",
        f"Model        {model} ({runtime}, device: {device})",
        f"Settings     {describe_settings(settings)}",
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
    backend: str = "pytorch",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The JSON document. ``details`` adds backend-specific facts (ONNX: execution providers)."""
    return {
        "edgevision_version": edgevision.__version__,
        "backend": backend,
        "image": image_path.as_posix(),
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "model": model,
        "device": device,
        **(details or {}),
        "settings": settings.to_dict(),
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
        settings = detection_settings(args, model_config)
    except ValueError as exc:
        parser.error(str(exc))  # exits with status 2
    model = model_name(args, model_config)
    try:
        check_model_matches_backend(args.backend, model)
        letterbox = pytorch_letterbox(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.json is not None and args.json.exists() and not args.overwrite:
        print(f"ERROR: {args.json} already exists; pass --overwrite to replace it.", file=sys.stderr)
        return 1

    try:
        image = read_image(args.image)
        if args.backend == "pytorch" and not Path(model).exists():
            print(f"Note: {model} not found locally; Ultralytics will try to download it.", file=sys.stderr)
        with create_detector(args.backend, model, device=args.device, settings=settings, letterbox=letterbox) as detector:
            device, runtime, details = detector.device, detector.description, detector_details(detector)
            detections = detector.detect(image, frame_index=0)
    except (ImageError, MissingDependencyError, DetectorError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    image_size = (image.shape[1], image.shape[0])
    print(format_summary(args.image, image_size, model, device, settings, detections, runtime))

    if args.json is not None:
        document = results_document(
            args.image, image_size, model, device, settings, detections, args.backend, details
        )
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
