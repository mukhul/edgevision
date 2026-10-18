"""Validate an ONNX model file and describe its inputs and outputs.

Run from the repository root:

    python scripts/inspect_onnx.py models/yolov8n.onnx
    python scripts/inspect_onnx.py models/yolov8n.onnx --imgsz 640

Runs the official ONNX checker and reports the opset, inputs, outputs and the
metadata Ultralytics stores. This is a structural check only: it does not
show that the model gives the same results as the PyTorch model.

Exit status: 0 valid, 1 invalid/missing file or missing package, 2 invalid arguments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from edgevision.cli import positive_int
from edgevision.models.onnx_validation import (
    OnnxModelInfo,
    OnnxValidationError,
    validate_onnx,
    yolo_layout_problems,
)
from edgevision.optional import MissingDependencyError


def format_report(info: OnnxModelInfo, problems: list[str]) -> str:
    lines = [
        "ONNX structural validation: PASSED (official ONNX checker, full check)",
        f"  File        {info.path} ({info.size_bytes / 2**20:.1f} MiB, {info.size_bytes:,} bytes)",
        f"  Opset       {info.opset if info.opset is not None else 'unknown'} (IR version {info.ir_version})",
        f"  Producer    {info.producer or 'unknown'}",
    ]
    lines += [f"  Input       {t.name}: {t.element_type} {t.shape_text}" for t in info.inputs]
    lines += [f"  Output      {t.name}: {t.element_type} {t.shape_text}" for t in info.outputs]
    names = info.class_names
    meta = info.metadata
    if meta:
        details = [f"task {meta['task']}" if "task" in meta else None,
                   f"imgsz {meta['imgsz']}" if "imgsz" in meta else None,
                   f"{len(names)} classes" if names is not None else None,
                   f"Ultralytics {meta['version']}" if "version" in meta else None]
        lines.append(f"  Metadata    {', '.join(d for d in details if d) or 'present'}")
    if problems:
        lines.append("  YOLO layout NOT as expected:")
        lines += [f"    - {p}" for p in problems]
    else:
        lines.append("  YOLO layout as expected: (batch, 3, H, W) in, (batch, 4 + classes, candidates) out")
    lines.append(
        "  Note: structural checks only. Matching PyTorch results requires a numerical comparison."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/inspect_onnx.py",
        description="Validate an ONNX model with the official checker and show its inputs and outputs.",
    )
    parser.add_argument("onnx", type=Path, help="ONNX file to check")
    parser.add_argument("--imgsz", type=positive_int, help="also check that the input is this square size")
    args = parser.parse_args(argv)
    try:
        info = validate_onnx(args.onnx)
    except (OnnxValidationError, MissingDependencyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    problems = yolo_layout_problems(info, args.imgsz)
    print(format_report(info, problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
