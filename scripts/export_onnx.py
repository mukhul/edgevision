"""Export the YOLO model from PyTorch to ONNX, then validate the ONNX file.

Run from the repository root:

    python scripts/export_onnx.py
    python scripts/export_onnx.py --model models/yolov8n.pt --output models/yolov8n.onnx --imgsz 640
    python scripts/export_onnx.py --overwrite

Defaults come from configs/default.yaml: model.weights (source),
model.onnx_path (output), model.input_size, export.opset,
export.dynamic_shapes and export.validate. Export runs on the CPU and never
downloads weights.

Exit status: 0 success, 1 export or validation failed (or missing package /
model / existing output), 2 invalid arguments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from edgevision.cli import positive_int
from edgevision.config import DEFAULT_CONFIG_PATH, ConfigError, load_config
from edgevision.models.export import ExportError, export_onnx
from edgevision.models.onnx_validation import OnnxValidationError, validate_onnx, yolo_layout_problems
from edgevision.optional import MissingDependencyError

EXAMPLES = """\
examples:
  python scripts/export_onnx.py
  python scripts/export_onnx.py --imgsz 320 --output models/yolov8n_320.onnx
  python scripts/export_onnx.py --overwrite
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/export_onnx.py",
        description="Export a YOLO .pt model to ONNX with Ultralytics (on the CPU) and validate the result.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=Path, help="source .pt model (default: model.weights from the config)")
    parser.add_argument("--output", type=Path, help="ONNX file to write (default: model.onnx_path from the config)")
    parser.add_argument("--imgsz", type=positive_int, help="input size, multiple of 32 (default: model.input_size)")
    parser.add_argument("--opset", type=positive_int, help="ONNX opset (default: export.opset; empty = Ultralytics' choice)")
    parser.add_argument(
        "--dynamic", action=argparse.BooleanOptionalAction, default=None,
        help="variable batch and image size in the graph (default: export.dynamic_shapes)",
    )
    parser.add_argument("--no-simplify", action="store_true", help="skip onnxslim graph simplification")
    parser.add_argument(
        "--validate", action=argparse.BooleanOptionalAction, default=None,
        help="run structural validation after export (default: export.validate)",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace the output file if it exists")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help=f"config file supplying defaults (default: {DEFAULT_CONFIG_PATH.as_posix()})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    source = args.model or config.model.weights
    output = args.output or config.model.onnx_path
    input_size = args.imgsz if args.imgsz is not None else config.model.input_size
    opset = args.opset if args.opset is not None else config.export.opset
    dynamic = args.dynamic if args.dynamic is not None else config.export.dynamic_shapes
    validate = args.validate if args.validate is not None else config.export.validate

    print(f"Exporting {source} -> {output} (input {input_size}, opset {opset or 'auto'}, "
          f"dynamic {dynamic}, simplify {not args.no_simplify}, CPU)")
    print("Ultralytics' log below mentions a temporary folder; the result is then moved to the path above.\n")
    try:
        result = export_onnx(
            source, output, input_size=input_size, opset=opset,
            simplify=not args.no_simplify, dynamic=dynamic, overwrite=args.overwrite,
        )
    except ValueError as exc:
        parser.error(str(exc))  # status 2
    except (ExportError, FileExistsError, MissingDependencyError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\nExported: {result.path} ({result.size_bytes / 2**20:.1f} MiB, {result.size_bytes:,} bytes)")
    if not validate:
        print("Validation skipped (--no-validate). Check later with: python scripts/inspect_onnx.py " + result.path.as_posix())
        return 0

    try:
        info = validate_onnx(result.path)
    except (OnnxValidationError, MissingDependencyError) as exc:
        print(f"ERROR: exported file failed validation: {exc}", file=sys.stderr)
        return 1
    problems = yolo_layout_problems(info, None if dynamic else input_size)
    io = ", ".join(f"{t.name} {t.shape_text}" for t in info.inputs + info.outputs)
    print(f"Validated: ONNX checker passed; opset {info.opset}; {io}")
    if problems:
        print("ERROR: the exported model does not have the expected YOLO layout:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print("Note: this is a structural check. It does not prove the ONNX model gives the same detections as PyTorch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
