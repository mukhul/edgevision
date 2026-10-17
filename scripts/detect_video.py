"""Run the PyTorch YOLO detector on every frame of a video and summarise the results.

Run from the repository root:

    python scripts/detect_video.py data/input/sample.mp4
    python scripts/detect_video.py data/input/sample.mp4 --max-frames 100 --json results/sample_detections.json

The video path, frame limit, model, input size, thresholds and class filter
default to configs/default.yaml; command-line options override them. This is
a processing summary, not a benchmark: no speed is measured or reported.

Exit status: 0 success, 1 the video/model/inference/output failed,
2 invalid arguments, 130 interrupted with Ctrl+C.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import ExitStack
from pathlib import Path

from edgevision.cli import (
    add_detector_arguments,
    describe_settings,
    detection_settings,
    model_name,
    positive_int,
)
from edgevision.config import ConfigError, load_config
from edgevision.models import DetectorError, PytorchYoloDetector
from edgevision.optional import MissingDependencyError
from edgevision.pipeline import PipelineError, PipelineSummary, VideoDetectionPipeline
from edgevision.results import VideoDetectionsWriter, video_header
from edgevision.video.reader import StopReason, VideoError

PROGRESS_EVERY = 25  # frames between progress lines on stderr

EXAMPLES = """\
examples:
  python scripts/detect_video.py data/input/sample.mp4
  python scripts/detect_video.py data/input/sample.mp4 --max-frames 100 --device cpu
  python scripts/detect_video.py data/input/sample.mp4 --all-frames --json results/sample_detections.json
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/detect_video.py",
        description="Detect objects in every frame of a video with the PyTorch YOLO backend.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "video", type=Path, nargs="?", help="video file (default: video.input from the config)"
    )
    limit = parser.add_mutually_exclusive_group()
    limit.add_argument(
        "--max-frames", type=positive_int, metavar="N",
        help="process at most N frames (default: video.max_frames from the config)",
    )
    limit.add_argument("--all-frames", action="store_true", help="process the whole video, ignoring the config limit")
    add_detector_arguments(parser)
    parser.add_argument("--json", type=Path, metavar="PATH", help="write per-frame detections and the summary to this JSON file")
    parser.add_argument("--overwrite", action="store_true", help="allow --json to replace an existing file")
    parser.add_argument("--quiet", action="store_true", help="do not print progress while processing")
    return parser


def format_summary(summary: PipelineSummary, model: str, device: str, settings_text: str) -> str:
    m = summary.metadata
    size = "unknown size" if m.width is None or m.height is None else f"{m.width} x {m.height} px"
    fps = "unknown" if m.fps is None else f"{m.fps:.3f}"
    reported = "unknown" if m.frame_count is None else str(m.frame_count)
    if summary.stop_reason is StopReason.MAX_FRAMES:
        stopped = f"frame limit ({summary.max_frames}) reached"
    else:
        stopped = "end of video"
    by_class = ", ".join(
        f"{name} {count}"
        for name, count in sorted(summary.detections_by_class.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    lines = [
        "Processing summary (not a benchmark: no timing is measured)",
        f"  Video                      {summary.video_path}",
        f"                             {size}, FPS {fps} (reported), {reported} frames (reported)",
        f"  Model                      {model} (PyTorch via Ultralytics, device: {device})",
        f"  Settings                   {settings_text}",
        f"  Frame limit                {summary.max_frames if summary.max_frames is not None else 'none'}",
        f"  Frames processed           {summary.frames_processed}",
        f"  Stopped because            {stopped}",
        f"  Frames with detections     {summary.frames_with_detections}",
        f"  Frames without detections  {summary.frames_without_detections}",
        f"  Total detections           {summary.total_detections}",
        f"  Detections by class        {by_class or 'none'}",
    ]
    if summary.ended_before_reported_count:
        lines.append(
            f"  Note: the file reported {m.frame_count} frames but only {summary.frames_processed} "
            "could be decoded."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    try:
        settings = detection_settings(args, config.model)
    except ValueError as exc:
        parser.error(str(exc))  # exits with status 2
    model = model_name(args, config.model)
    video = args.video if args.video is not None else config.video.input
    if args.all_frames:
        max_frames = None
    else:
        max_frames = args.max_frames if args.max_frames is not None else config.video.max_frames

    # Cheap checks first, so a typo does not cost a model load.
    if not video.is_file():
        reason = "not found" if not video.exists() else "not a file"
        print(f"ERROR: Video {reason}: {video} (looked relative to {Path.cwd()})", file=sys.stderr)
        return 1
    if args.json is not None and args.json.exists() and not args.overwrite:
        print(f"ERROR: {args.json} already exists; pass --overwrite to replace it.", file=sys.stderr)
        return 1

    if not Path(model).exists():
        print(f"Note: {model} not found locally; Ultralytics will try to download it.", file=sys.stderr)
    try:
        # ExitStack closes everything in reverse order when the block ends, even on
        # errors: the results writer first (deleting its .partial file unless it
        # finished), then the video, then the detector.
        with ExitStack() as stack:
            detector = stack.enter_context(
                PytorchYoloDetector(model, device=args.device, settings=settings)
            )
            pipeline = stack.enter_context(
                VideoDetectionPipeline(video, detector, max_frames=max_frames)
            )
            writer = None
            if args.json is not None:
                header = video_header(
                    metadata=pipeline.metadata,
                    backend=detector.backend_name,
                    model=model,
                    device=detector.device,
                    settings=settings.to_dict(),
                )
                writer = stack.enter_context(
                    VideoDetectionsWriter(args.json, header, overwrite=args.overwrite)
                )
            for done, result in enumerate(pipeline, start=1):
                if writer is not None:
                    writer.write_frame(result)
                if not args.quiet and done % PROGRESS_EVERY == 0:
                    print(f"  processed {done} frames...", file=sys.stderr)
            summary = pipeline.summary()
            if writer is not None:
                writer.finish(summary.to_dict())
            device = detector.device
    except KeyboardInterrupt:
        print("\nInterrupted: no results file was written.", file=sys.stderr)
        return 130
    except (VideoError, PipelineError, DetectorError, MissingDependencyError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if args.json is not None:
            print("No results file was written.", file=sys.stderr)
        return 1

    print(format_summary(summary, model, device, describe_settings(settings)))
    if args.json is not None:
        print(f"\nSaved JSON: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
