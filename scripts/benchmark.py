"""Benchmark a detection backend and save the measurements as JSON.

Run from the repository root:

    python scripts/benchmark.py --backend pytorch --device cpu
    python scripts/benchmark.py --backend onnx --device cpu
    python scripts/benchmark.py data/input/sample.mp4 --backend onnx --scope pipeline --tracking

Scopes (see docs/benchmarking.md for the exact timing boundaries):

  detector  (default) frames are decoded into memory first; each iteration times one
            detect() call: preprocessing + inference + postprocessing. A video
            contributes its first --preload-frames frames, used in rotation.
  pipeline  video only; each iteration reads the next frame and runs the detector
            (and ByteTrack with --tracking). Decoding, detection and tracking are
            timed separately. Drawing and file writing are never timed.

Every run creates a new detector (initialisation is timed separately), runs
--warmup untimed iterations, then --iterations measured ones. Defaults come from
the `benchmark` and `model` sections of configs/default.yaml. Models are never
downloaded. Results go to results/benchmarks/ unless --output is given, and an
existing file is only replaced with --overwrite.

Exit status: 0 success, 1 benchmark failed (missing model/input/package, error
during a run, output exists), 2 invalid arguments, 130 interrupted.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from edgevision.benchmarks.report import (
    build_report,
    environment_snapshot,
    file_identity,
    format_summary,
    git_revision,
    write_report,
)
from edgevision.benchmarks.runner import (
    BenchmarkError,
    BenchmarkPlan,
    model_input_shape,
    run_benchmark,
    time_library_imports,
)
from edgevision.cli import add_detector_arguments, detection_settings, model_name, positive_int
from edgevision.config import BENCHMARK_SCOPES, ConfigError, load_config
from edgevision.models import DetectorError, create_detector
from edgevision.models.factory import check_model_matches_backend
from edgevision.optional import MissingDependencyError, import_optional
from edgevision.tracking import TrackingError
from edgevision.video.reader import VideoError, VideoReader

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_OUTPUT_DIR = Path("results") / "benchmarks"
MIN_SAMPLES_FOR_PERCENTILES = 20

EXAMPLES = """\
examples:
  python scripts/benchmark.py --backend pytorch --device cpu
  python scripts/benchmark.py --backend onnx --device cpu
  python scripts/benchmark.py data/input/sample.mp4 --backend onnx --warmup 10 --iterations 100 --runs 3
  python scripts/benchmark.py photo.jpg --backend pytorch --output results/benchmarks/pt_photo.json
  python scripts/benchmark.py data/input/sample.mp4 --scope pipeline --tracking --backend onnx
  python scripts/compare_benchmarks.py results/benchmarks/a.json results/benchmarks/b.json
"""


def non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be an integer >= 0, got {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be an integer >= 0, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/benchmark.py",
        description="Measure initialisation time, per-frame latency and throughput of a detection backend.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, nargs="?", help="image or video file (default: video.input from the config)")
    add_detector_arguments(parser)
    parser.set_defaults(device=None)  # default from benchmark.device in the config
    parser.add_argument("--scope", choices=list(BENCHMARK_SCOPES), help="what is timed (default: benchmark.scope)")
    parser.add_argument("--warmup", type=non_negative_int, metavar="N",
                        help="untimed iterations per run before measuring (default: benchmark.warmup)")
    parser.add_argument("--iterations", type=positive_int, metavar="N",
                        help="measured iterations (frames) per run (default: benchmark.iterations)")
    parser.add_argument("--runs", type=positive_int, metavar="N",
                        help="independent runs, each with a newly loaded model (default: benchmark.runs)")
    parser.add_argument("--preload-frames", type=positive_int, metavar="N",
                        help="detector scope: video frames decoded into memory and used in rotation "
                        "(default: benchmark.preload_frames)")
    parser.add_argument("--tracking", action="store_true", help="pipeline scope: also time ByteTrack per frame")
    parser.add_argument(
        "--pytorch-letterbox", choices=["square", "minimal"],
        help="PyTorch only: 'square' (default) pads frames to the full input size like the ONNX model, so both "
        "backends process the same tensor; 'minimal' is Ultralytics' normal padding to a multiple of 32",
    )
    parser.add_argument("--output", type=Path, metavar="PATH",
                        help=f"result JSON file (default: {DEFAULT_OUTPUT_DIR.as_posix()}/<backend>_<device>_<scope>_<time>.json)")
    parser.add_argument("--overwrite", action="store_true", help="allow --output to replace an existing file")
    parser.add_argument("--quiet", action="store_true", help="do not print progress")
    return parser


# --- inputs --------------------------------------------------------------------------------


def read_image(path: Path) -> Any:
    cv2 = import_optional("cv2", display_name="OpenCV", extra="vision", purpose="read images")
    import numpy as np

    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise BenchmarkError(f"Could not decode {path} as an image")
    return image


def load_input(path: Path, kind: str, plan: BenchmarkPlan) -> tuple[list[Any] | None, dict[str, Any]]:
    """Frames for the detector scope (None for the pipeline scope) and a description of the input."""
    info: dict[str, Any] = {"kind": kind, "file": file_identity(path)}
    if kind == "image":
        image = read_image(path)
        info.update(width=image.shape[1], height=image.shape[0], fps=None, frame_count_reported=None,
                    frames_in_memory=1)
        return [image], info
    with VideoReader(path, max_frames=plan.preload_frames if plan.scope == "detector" else 1) as reader:
        m = reader.metadata
        frames = [frame for _, frame in reader]
    if not frames:
        raise BenchmarkError(f"No frames could be decoded from {path}")
    info.update(width=frames[0].shape[1], height=frames[0].shape[0], fps=m.fps,
                frame_count_reported=m.frame_count,
                frames_in_memory=len(frames) if plan.scope == "detector" else None)
    return (frames if plan.scope == "detector" else None), info


def default_output(backend: str, device: str, scope: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"{backend}_{device}_{scope}_{stamp}.json"


# --- main ----------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    bench = config.benchmark

    # Argument checks (status 2) before anything slow.
    try:
        settings = detection_settings(args, config.model)
        plan = BenchmarkPlan(
            scope=args.scope or bench.scope,
            warmup=args.warmup if args.warmup is not None else bench.warmup,
            iterations=args.iterations or bench.iterations,
            runs=args.runs or bench.runs,
            preload_frames=args.preload_frames or bench.preload_frames,
            tracking=args.tracking,
        )
        model = model_name(args, config.model)
        check_model_matches_backend(args.backend, model)
    except ValueError as exc:
        parser.error(str(exc))
    if args.backend == "onnx" and args.pytorch_letterbox == "minimal":
        parser.error("--pytorch-letterbox minimal applies only to --backend pytorch (ONNX models need a square input)")
    letterbox = "square" if args.backend == "onnx" else (args.pytorch_letterbox or "square")
    device = args.device or bench.device
    source = args.input if args.input is not None else config.video.input
    kind = "image" if source.suffix.lower() in IMAGE_SUFFIXES else "video"
    if plan.scope == "pipeline" and kind == "image":
        parser.error("the pipeline scope needs a video input (it times video decoding)")
    output = args.output or default_output(args.backend, device, plan.scope)
    if output.suffix.lower() != ".json":
        parser.error(f"--output must end in .json, got {output}")

    # File checks (status 1), still before loading anything.
    for label, path in (("Input", source), ("Model", Path(model))):
        if not path.is_file():
            hint = ""
            if label == "Model":
                hint = (" Benchmarks never download models: run scripts/detect_image.py once for yolov8n.pt"
                        if args.backend == "pytorch" else " Create it with: python scripts/export_onnx.py")
            print(f"ERROR: {label} file not found: {path} (looked relative to {Path.cwd()}).{hint}", file=sys.stderr)
            return 1
    if output.exists() and not args.overwrite:
        print(f"ERROR: {output} already exists; pass --overwrite to replace it.", file=sys.stderr)
        return 1

    def say(message: str) -> None:
        if not args.quiet:
            print(f"  {message}", file=sys.stderr)

    warnings: list[str] = []
    created: list[Any] = []

    def make_detector() -> Any:
        detector = create_detector(args.backend, model, device=device, settings=settings, letterbox=letterbox)
        created.append(detector)
        return detector

    make_tracker = None
    if plan.tracking:
        from edgevision.tracking import ByteTrackTracker

        make_tracker = ByteTrackTracker

    try:
        say(f"loading {kind} {source}")
        frames, input_info = load_input(source, kind, plan)
        say(f"importing {args.backend} libraries")
        library_import = time_library_imports(args.backend)
        runs = run_benchmark(
            plan, make_detector, frames=frames,
            open_video=(lambda: VideoReader(source)) if plan.scope == "pipeline" else None,
            make_tracker=make_tracker, progress=say,
        )
    except KeyboardInterrupt:
        print("\nInterrupted: no results were written.", file=sys.stderr)
        return 130
    except (BenchmarkError, DetectorError, MissingDependencyError, VideoError, TrackingError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    detector = created[-1]
    devices = {(r.device, tuple(r.execution_providers or ())) for r in runs}
    if len(devices) > 1:
        warnings.append(f"runs used different devices/providers: {sorted(devices)}")
    if device == "auto" and runs[0].device == "cpu":
        warnings.append("device 'auto' selected the CPU (no usable CUDA device/provider): this is a CPU result")
    if plan.iterations < MIN_SAMPLES_FOR_PERCENTILES:
        warnings.append(f"only {plan.iterations} measured iterations per run: p90/p95 are not reliable")
    if letterbox == "minimal":
        warnings.append("PyTorch used minimal padding: its input tensor differs from the ONNX backend's square input")

    backend_info = {
        "name": args.backend,
        "runtime": getattr(detector, "description", args.backend),
        "model": file_identity(model),
        "requested_device": device,
        "device": runs[0].device,
        "execution_providers": runs[0].execution_providers,
        "letterbox": letterbox,
        "model_input_shape": model_input_shape(input_info["height"], input_info["width"], settings.input_size, letterbox),
        "settings": settings.to_dict(),
        "precision": "fp32",
    }
    document = build_report(
        plan=plan.to_dict(), backend=backend_info, input_info=input_info, runs=runs,
        library_import=library_import, environment=environment_snapshot(args.backend),
        git=git_revision(), warnings=warnings,
    )
    try:
        write_report(output, document, overwrite=args.overwrite)
    except (OSError, ValueError) as exc:
        print(f"ERROR: could not write {output}: {exc}", file=sys.stderr)
        return 1
    print(format_summary(document))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
