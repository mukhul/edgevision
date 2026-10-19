"""Run a YOLO detector on every frame of a video, summarise the results,
and optionally save per-frame JSON and/or an annotated copy of the video.

Run from the repository root:

    python scripts/detect_video.py data/input/sample.mp4
    python scripts/detect_video.py data/input/sample.mp4 --max-frames 100 --json results/sample_detections.json
    python scripts/detect_video.py data/input/sample.mp4 --output-video results/sample_annotated.mp4
    python scripts/detect_video.py data/input/sample.mp4 --tracking --output-video results/sample_tracked.mp4
    python scripts/detect_video.py data/input/sample.mp4 --backend onnx --device cpu

--backend pytorch (the default) runs the .pt model with Ultralytics;
--backend onnx runs an exported .onnx model with ONNX Runtime. Everything
after detection (tracking, annotation, JSON, video output) is shared. The video path, frame limit, model, input size, thresholds and class filter
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
    detector_details,
    model_name,
    pytorch_letterbox,
    positive_float,
    positive_int,
)
from edgevision.config import ConfigError, load_config
from edgevision.models import DetectorError, create_detector
from edgevision.models.factory import check_model_matches_backend
from edgevision.optional import MissingDependencyError
from edgevision.pipeline import PipelineError, PipelineSummary, VideoDetectionPipeline
from edgevision.results import VideoDetectionsWriter, video_header
from edgevision.tracking import ByteTrackTracker, TrackingError
from edgevision.video.annotator import annotate_frame
from edgevision.video.reader import StopReason, VideoError
from edgevision.video.writer import (
    DEFAULT_CODEC,
    FALLBACK_FPS,
    VideoWriteError,
    VideoWriter,
    choose_output_fps,
)

PROGRESS_EVERY = 25  # frames between progress lines on stderr

EXAMPLES = """\
examples:
  python scripts/detect_video.py data/input/sample.mp4
  python scripts/detect_video.py data/input/sample.mp4 --max-frames 100 --device cpu
  python scripts/detect_video.py data/input/sample.mp4 --all-frames --json results/sample_detections.json
  python scripts/detect_video.py data/input/sample.mp4 --output-video results/sample_annotated.mp4
  python scripts/detect_video.py data/input/sample.mp4 --tracking --json results/sample_tracks.json
  python scripts/detect_video.py data/input/sample.mp4 --backend onnx --json results/sample_onnx.json
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/detect_video.py",
        description="Detect objects in every frame of a video with a YOLO model on PyTorch or ONNX Runtime.",
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
    parser.add_argument(
        "--output-video", type=Path, metavar="PATH",
        help=f"write an annotated copy of the processed frames to this .mp4 file (codec {DEFAULT_CODEC})",
    )
    parser.add_argument(
        "--fps", type=positive_float, metavar="FPS",
        help="frame rate of --output-video (default: the input video's FPS; "
        f"{FALLBACK_FPS:g} with a warning if the input does not report one)",
    )
    parser.add_argument(
        "--tracking", action=argparse.BooleanOptionalAction, default=None,
        help="track objects across frames with ByteTrack and add track IDs to the output "
        "(default: tracking.enabled from the config, which is off)",
    )
    parser.add_argument(
        "--tracker-config", type=Path, metavar="YAML",
        help="ByteTrack settings file in Ultralytics' bytetrack.yaml format (default: Ultralytics' built-in one)",
    )
    parser.add_argument("--overwrite", action="store_true", help="allow --json / --output-video to replace existing files")
    parser.add_argument("--quiet", action="store_true", help="do not print progress while processing")
    return parser


def same_file(a: Path, b: Path) -> bool:
    """True if two paths point to the same file (works for existing and new paths)."""
    try:
        if a.exists() and b.exists():
            return a.samefile(b)
    except OSError:
        pass
    return a.resolve() == b.resolve()


def format_summary(
    summary: PipelineSummary, model: str, device: str, settings_text: str, runtime: str = "PyTorch via Ultralytics"
) -> str:
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
        f"  Model                      {model} ({runtime}, device: {device})",
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
    try:
        check_model_matches_backend(args.backend, model)
        letterbox = pytorch_letterbox(args)
    except ValueError as exc:
        parser.error(str(exc))
    video = args.video if args.video is not None else config.video.input
    if args.all_frames:
        max_frames = None
    else:
        max_frames = args.max_frames if args.max_frames is not None else config.video.max_frames

    tracking = args.tracking if args.tracking is not None else config.tracking.enabled

    # Argument conflicts (status 2), then cheap file checks (status 1), all before loading a model.
    if args.tracker_config is not None and not tracking:
        parser.error("--tracker-config only applies when tracking is enabled (--tracking)")
    out_video = args.output_video
    if args.fps is not None and out_video is None:
        parser.error("--fps only applies together with --output-video")
    if out_video is not None:
        if out_video.suffix.lower() != ".mp4":
            parser.error(f"--output-video must end in .mp4, got {out_video}")
        if same_file(out_video, video):
            parser.error("--output-video must not be the input video")
        if args.json is not None and same_file(out_video, args.json):
            parser.error("--output-video and --json must be different files")
    if not video.is_file():
        reason = "not found" if not video.exists() else "not a file"
        print(f"ERROR: Video {reason}: {video} (looked relative to {Path.cwd()})", file=sys.stderr)
        return 1
    for output in (args.json, out_video):
        if output is not None and output.exists() and not args.overwrite:
            print(f"ERROR: {output} already exists; pass --overwrite to replace it.", file=sys.stderr)
            return 1

    if args.backend == "pytorch" and not Path(model).exists():
        print(f"Note: {model} not found locally; Ultralytics will try to download it.", file=sys.stderr)
    tracker = None
    if tracking:
        try:  # cheap, and fails fast on a missing package or a bad config
            tracker = ByteTrackTracker(args.tracker_config)
        except (TrackingError, MissingDependencyError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    video_writer = None
    try:
        # ExitStack closes everything in reverse order when the block ends, even on
        # errors: output writers first (deleting their partial files unless they
        # finished), then the input video, then the detector.
        with ExitStack() as stack:
            detector = stack.enter_context(
                create_detector(args.backend, model, device=args.device, settings=settings, letterbox=letterbox)
            )
            pipeline = stack.enter_context(
                VideoDetectionPipeline(video, detector, max_frames=max_frames, tracker=tracker)
            )
            json_writer = None
            if args.json is not None:
                header = video_header(
                    metadata=pipeline.metadata,
                    backend=detector.backend_name,
                    model=model,
                    device=detector.device,
                    settings=settings.to_dict(),
                    details=detector_details(detector),
                    tracker=None if tracker is None else {
                        "name": tracker.name, "config": tracker.config_source, "settings": tracker.settings,
                    },
                )
                json_writer = stack.enter_context(
                    VideoDetectionsWriter(args.json, header, overwrite=args.overwrite)
                )
            if out_video is not None:
                out_fps, fps_source = choose_output_fps(pipeline.metadata.fps, args.fps)
                if fps_source == "fallback":
                    print(
                        f"WARNING: {video} does not report a usable frame rate; writing the annotated "
                        f"video at {out_fps:g} FPS. Its playback speed may be wrong; pass --fps to set it.",
                        file=sys.stderr,
                    )

            for done, result in enumerate(pipeline, start=1):
                if json_writer is not None:
                    json_writer.write_frame(result)
                if out_video is not None:
                    if video_writer is None:
                        # Open on the first frame so the output uses the size of the frames
                        # actually decoded (it can differ from the header, e.g. rotated phone video).
                        height, width = result.image.shape[:2]
                        video_writer = stack.enter_context(
                            VideoWriter(out_video, fps=out_fps, frame_size=(width, height), overwrite=args.overwrite)
                        )
                    video_writer.write(annotate_frame(result.image, result.detections))
                if not args.quiet and done % PROGRESS_EVERY == 0:
                    print(f"  processed {done} frames...", file=sys.stderr)

            summary = pipeline.summary()
            # Finish the video first: if that fails, the JSON is discarded too, so the
            # two outputs never disagree about whether the run succeeded.
            if video_writer is not None:
                video_writer.finish()
            if json_writer is not None:
                json_writer.finish(summary.to_dict())
            device, runtime = detector.device, detector.description
    except KeyboardInterrupt:
        print("\nInterrupted: no output files were written.", file=sys.stderr)
        return 130
    except (VideoError, PipelineError, DetectorError, MissingDependencyError, VideoWriteError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if args.json is not None or out_video is not None:
            print("No output files were written.", file=sys.stderr)
        return 1

    print(format_summary(summary, model, device, describe_settings(settings), runtime))
    if tracker is not None:
        print(
            f"  Tracking                   {tracker.name} ({tracker.config_source})\n"
            f"  Detections with a track ID {summary.detections_with_track_id} of {summary.total_detections}\n"
            f"  Distinct track IDs         {summary.distinct_track_ids} observed in this video "
            "(not a count of real objects: IDs can be lost and reassigned)"
        )
    if video_writer is not None:
        print(
            f"  Annotated video            {out_video}\n"
            f"                             {video_writer.width} x {video_writer.height} px, "
            f"{video_writer.frames_written} frames, {video_writer.fps:g} FPS ({fps_source}), "
            f"codec {video_writer.codec}"
        )
    if args.json is not None:
        print(f"\nSaved JSON: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
