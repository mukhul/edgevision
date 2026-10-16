"""Inspect a video: report its metadata and how many frames can actually be decoded.

Run from the repository root:

    python scripts/inspect_video.py data/input/sample.mp4
    python scripts/inspect_video.py data/input/sample.mp4 --max-frames 100

Exit status: 0 on success, 1 if the video cannot be read, 2 for invalid arguments.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from edgevision.video.reader import (
    MissingDependencyError,
    StopReason,
    VideoError,
    VideoMetadata,
    VideoReader,
)

EXAMPLES = """\
examples:
  python scripts/inspect_video.py data/input/sample.mp4
  python scripts/inspect_video.py data/input/sample.mp4 --max-frames 100
"""


def positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def known(value: object, fmt: str = "{}") -> str:
    return "unknown" if value is None else fmt.format(value)


def build_report(
    metadata: VideoMetadata,
    frames_read: int,
    stop_reason: StopReason | None,
    max_frames: int | None,
    decoded_size: tuple[int, int] | None,
    ended_early: bool,
) -> str:
    m = metadata
    size = "unknown" if m.width is None or m.height is None else f"{m.width} x {m.height} px"
    lines = [
        f"Video                {m.path}",
        f"Resolution           {size} (reported)",
        f"FPS                  {known(m.fps, '{:.3f}')} (reported)",
        f"Frame count          {known(m.frame_count)} (reported)",
        f"Duration             {known(m.duration_s, '{:.2f} s')} (reported frame count / FPS)",
        f"Codec (FourCC)       {known(m.fourcc)}",
        "",
        f"Frames read          {frames_read}",
    ]
    if stop_reason is StopReason.MAX_FRAMES:
        lines.append(f"Stopped because      --max-frames limit ({max_frames}) was reached")
    else:
        lines.append("Stopped because      end of video (EOF) was reached")

    notes = []
    if decoded_size is not None and (m.width, m.height) != decoded_size:
        notes.append(
            f"Decoded frames are {decoded_size[0]} x {decoded_size[1]} px, not the reported size "
            "(for example because of rotation metadata in phone videos)."
        )
    if ended_early:
        notes.append(
            f"The file reported {m.frame_count} frames but only {frames_read} could be read. "
            "It may be truncated, or the reported count may be inaccurate."
        )
    elif (
        stop_reason is StopReason.END_OF_VIDEO
        and m.frame_count is not None
        and frames_read > m.frame_count
    ):
        notes.append("More frames were read than the file reported; its frame count is inaccurate.")
    if notes:
        lines += [""] + [f"Note: {n}" for n in notes]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/inspect_video.py",
        description="Report a video's metadata and count the frames that can actually be decoded.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video", type=Path, help="path to the video file")
    parser.add_argument(
        "--max-frames",
        type=positive_int,
        default=None,
        metavar="N",
        help="stop after reading N frames (default: read the whole video)",
    )
    args = parser.parse_args(argv)
    # This script prints its own notes, so only show log messages that are errors.
    logging.basicConfig(level=logging.ERROR)

    decoded_size = None
    try:
        with VideoReader(args.video, max_frames=args.max_frames) as reader:
            for _, frame in reader:
                if decoded_size is None:
                    height, width = frame.shape[:2]
                    decoded_size = (width, height)
    except (VideoError, MissingDependencyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        build_report(
            reader.metadata,
            reader.frames_read,
            reader.stop_reason,
            args.max_frames,
            decoded_size,
            reader.ended_before_reported_count,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
