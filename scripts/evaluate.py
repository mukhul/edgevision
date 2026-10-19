"""Evaluate detection accuracy against labelled images (COCO JSON ground truth).

Run from the repository root:

    python scripts/evaluate.py --annotations data/eval/annotations.json --images data/eval/images
    python scripts/evaluate.py --annotations data/eval/annotations.json --images data/eval/images --backend onnx --conf 0.001

Runs the chosen backend on every labelled image and reports precision,
recall and F1 (IoU 0.50, at the detector's confidence threshold) and COCO-style
AP / mAP@0.5 and mAP@0.5:0.95, overall and per class. The annotation format
is documented in docs/evaluation.md. This repository ships no labelled data:
these numbers exist only for a dataset you provide.

Exit status: 0 success, 1 bad dataset/model/output, 2 invalid arguments, 130 interrupted.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import edgevision
from edgevision.benchmarks.report import display_path, file_identity
from edgevision.cli import add_detector_arguments, detection_settings, detector_details, model_name, pytorch_letterbox
from edgevision.config import ConfigError, load_config
from edgevision.models import DetectorError, create_detector
from edgevision.models.factory import check_model_matches_backend
from edgevision.optional import MissingDependencyError, import_optional

DEFAULT_OUTPUT_DIR = Path("results") / "evaluation"
NOTES = [
    "Precision, recall and F1 use all detections returned at the detector's confidence threshold, matched at IoU 0.50.",
    "AP/mAP are computed from the same detections. With a high confidence threshold (default 0.25) low-confidence "
    "detections are missing, so AP is lower than published COCO numbers, which use 0.001. Use --conf 0.001 to compare.",
    "Classes without ground truth in this dataset are not part of mAP; detections of classes absent from the dataset's "
    "categories are counted separately and not evaluated.",
    "Results describe this dataset only. A small or unrepresentative dataset gives uncertain numbers.",
]


class EvaluationError(Exception):
    """The evaluation could not be completed."""


def read_image(path: Path) -> Any:
    cv2 = import_optional("cv2", display_name="OpenCV", extra="vision", purpose="read images")
    import numpy as np

    if not path.is_file():
        raise EvaluationError(f"Image listed in the annotations not found: {path}")
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise EvaluationError(f"Could not decode {path} as an image")
    return image


def with_names(metrics: dict[str, Any], class_names: dict[int, str]) -> dict[str, Any]:
    out = dict(metrics)
    out["per_class"] = {class_names[c]: {"class_id": c, **v} for c, v in metrics["per_class"].items()}
    out["classes_without_ground_truth"] = [class_names[c] for c in metrics["classes_without_ground_truth"]]
    return out


def _f(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:.3f}"


def format_summary(doc: dict[str, Any]) -> str:
    m, ds, b = doc["metrics"], doc["dataset"], doc["backend"]
    o = m["overall"]
    lines = [
        "Detection accuracy on the supplied labelled dataset",
        f"  Backend      {b['name']} ({b['model']['path']}, device {b['device']})",
        f"  Dataset      {ds['annotations']['path']}: {ds['images']} images, {ds['objects']} labelled objects, "
        f"{len(ds['classes'])} classes",
        f"  Settings     conf {b['settings']['conf_threshold']}, NMS IoU {b['settings']['iou_threshold']}, "
        f"input {b['settings']['input_size']}",
        "",
        f"  Overall      precision {_f(o['precision'])}  recall {_f(o['recall'])}  F1 {_f(o['f1'])}  "
        f"mAP@0.5 {_f(o['map50'])}  mAP@0.5:0.95 {_f(o['map50_95'])}",
        f"               TP {o['tp']}  FP {o['fp']}  FN {o['fn']}  (IoU 0.50)",
        "",
        f"  {'class':<16}{'objects':>8}{'preds':>7}{'P':>7}{'R':>7}{'F1':>7}{'AP50':>7}{'AP50:95':>9}",
    ]
    for name, c in m["per_class"].items():
        lines.append(f"  {name:<16}{c['ground_truth']:>8}{c['predictions']:>7}{_f(c['precision']):>7}"
                     f"{_f(c['recall']):>7}{_f(c['f1']):>7}{_f(c['ap50']):>7}{_f(c['ap50_95']):>9}")
    lines += [f"  Note: {n}" for n in list(ds["warnings"]) + NOTES[:2]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/evaluate.py",
        description="Measure detection accuracy against labelled images in COCO JSON format.",
    )
    parser.add_argument("--annotations", type=Path, required=True, help="COCO JSON file with the ground truth")
    parser.add_argument("--images", type=Path, required=True, help="folder containing the annotated images")
    add_detector_arguments(parser)
    parser.add_argument("--output", type=Path, metavar="PATH",
                        help=f"result JSON (default: {DEFAULT_OUTPUT_DIR.as_posix()}/<backend>_<date-time>.json)")
    parser.add_argument("--overwrite", action="store_true", help="allow --output to replace an existing file")
    args = parser.parse_args(argv)
    try:  # imported here so that --help and argument errors work without NumPy
        from edgevision.evaluation.annotations import AnnotationError, load_coco
        from edgevision.evaluation.detection_metrics import evaluate_detections
    except MissingDependencyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    try:
        settings = detection_settings(args, config.model)
        model = model_name(args, config.model)
        check_model_matches_backend(args.backend, model)
        letterbox = pytorch_letterbox(args)
    except ValueError as exc:
        parser.error(str(exc))
    output = args.output or DEFAULT_OUTPUT_DIR / f"{args.backend}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    if output.suffix.lower() != ".json":
        parser.error(f"--output must end in .json, got {output}")
    for label, path in (("Annotation file", args.annotations), ("Model file", Path(model))):
        if not path.is_file():
            print(f"ERROR: {label} not found: {path} (evaluation never downloads anything)", file=sys.stderr)
            return 1
    if not args.images.is_dir():
        print(f"ERROR: Images folder not found: {args.images}", file=sys.stderr)
        return 1
    if output.exists() and not args.overwrite:
        print(f"ERROR: {output} already exists; pass --overwrite to replace it.", file=sys.stderr)
        return 1

    try:
        with create_detector(args.backend, model, device=args.device, settings=settings, letterbox=letterbox) as detector:
            dataset = load_coco(args.annotations, detector.class_names)
            predictions = {}
            for index, image in enumerate(dataset.images):
                frame = read_image(args.images / image.file_name)
                if (frame.shape[1], frame.shape[0]) != (image.width, image.height):
                    raise EvaluationError(
                        f"{image.file_name} is {frame.shape[1]} x {frame.shape[0]} px but the annotations say "
                        f"{image.width} x {image.height}; the boxes would not line up"
                    )
                predictions[image.image_id] = detector.detect(frame, frame_index=index)
            device, details, description = detector.device, detector_details(detector), detector.description
    except KeyboardInterrupt:
        print("\nInterrupted: no results were written.", file=sys.stderr)
        return 130
    except (AnnotationError, EvaluationError, DetectorError, MissingDependencyError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    metrics = evaluate_detections(dataset.boxes, predictions, list(dataset.class_names))
    document = {
        "schema": "edgevision.evaluation",
        "schema_version": 1,
        "edgevision_version": edgevision.__version__,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "backend": {"name": args.backend, "runtime": description, "model": file_identity(model), "device": device,
                    **details, "settings": settings.to_dict()},
        "dataset": {
            "annotations": file_identity(args.annotations),
            "images_folder": display_path(args.images),
            "images": len(dataset.images),
            "objects": dataset.num_objects,
            "classes": dataset.class_names,
            "unmapped_categories": list(dataset.unmapped_categories),
            "warnings": list(dataset.warnings),
        },
        "metrics": with_names(metrics, detector.class_names),
        "notes": NOTES,
    }
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(output.name + ".partial")
        partial.write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")
        partial.replace(output)
    except (OSError, ValueError) as exc:
        print(f"ERROR: could not write {output}: {exc}", file=sys.stderr)
        return 1
    print(format_summary(document))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
