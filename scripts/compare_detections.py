"""Compare the detections of two detect_video.py runs on the same video.

Run from the repository root:

    python scripts/compare_detections.py results/sample_detections.json results/sample_onnx.json
    python scripts/compare_detections.py a.json b.json --iou 0.7 --output results/analysis/agreement.json

Matches detections frame by frame (same class, IoU >= --iou) and reports how
many agree, which ones only one run found, and how close matched boxes and
confidences are. This measures agreement between two runs (e.g. PyTorch vs
ONNX Runtime), NOT accuracy: there is no ground truth.

Exit status: 0 success, 1 unusable file or output exists, 2 invalid arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from edgevision.cli import fraction
from edgevision.optional import MissingDependencyError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/compare_detections.py",
        description="How closely two detect_video.py --json outputs agree (not an accuracy measure).",
    )
    parser.add_argument("a", type=Path, help="first detections JSON (A)")
    parser.add_argument("b", type=Path, help="second detections JSON (B), same video")
    parser.add_argument("--iou", type=fraction, default=0.5, help="IoU needed for two boxes to match (default 0.5)")
    parser.add_argument("--output", type=Path, metavar="PATH", help="also save the comparison as JSON")
    parser.add_argument("--overwrite", action="store_true", help="allow --output to replace an existing file")
    args = parser.parse_args(argv)
    if args.iou == 0:
        parser.error("--iou must be greater than 0")
    try:  # imported here so that --help and argument errors work without NumPy
        from edgevision.evaluation.agreement import DetectionsFileError, compare_files, format_comparison
    except MissingDependencyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.output is not None and args.output.exists() and not args.overwrite:
        print(f"ERROR: {args.output} already exists; pass --overwrite to replace it.", file=sys.stderr)
        return 1
    try:
        report = compare_files(args.a, args.b, iou_threshold=args.iou)
    except DetectionsFileError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(format_comparison(report))
    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"schema": "edgevision.detection_agreement", "schema_version": 1,
                                               **report}, indent=2), encoding="utf-8")
        except OSError as exc:
            print(f"ERROR: could not write {args.output}: {exc}", file=sys.stderr)
            return 1
        print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
