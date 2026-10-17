"""Write frames to a video file with OpenCV, safely.

Usage::

    from edgevision.video.writer import VideoWriter

    with VideoWriter("results/annotated.mp4", fps=30.0, frame_size=(1024, 576)) as writer:
        for frame in frames:          # BGR uint8 arrays, all exactly 1024 x 576
            writer.write(frame)
        writer.finish()               # only now does results/annotated.mp4 appear

Frames go to ``<name>.partial<ext>`` (for example ``annotated.partial.mp4``;
the real extension is kept so OpenCV picks the right container). ``finish()``
closes the file and renames it to the requested name. If the ``with`` block
ends without ``finish()``, because of an error or Ctrl+C, the partial file is
deleted, so an interrupted run never leaves a video that looks finished.

The default codec is ``mp4v`` (MPEG-4 Part 2) in an ``.mp4`` file. It is
available in the standard ``opencv-python`` wheels, whereas H.264 (``avc1``)
usually is not. It plays in VLC and in common desktop players; some web
browsers cannot play it. Width and height must be even: OpenCV's encoders
silently drop the last row or column of odd-sized frames, so odd sizes are
rejected instead.
"""

from __future__ import annotations

import logging
import math
import numbers
import os
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

from edgevision.models.base import validate_bgr_frame
from edgevision.optional import import_optional

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CODEC = "mp4v"
#: Used only when the source video does not report a usable FPS and none is given.
FALLBACK_FPS = 30.0


class VideoWriteError(Exception):
    """The output video could not be opened or written."""


def _is_integer(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def validate_fps(fps: Any) -> float:
    if isinstance(fps, bool) or not isinstance(fps, numbers.Real) or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be a finite number greater than 0, got {fps!r}")
    return float(fps)


def choose_output_fps(source_fps: float | None, override: float | None = None) -> tuple[float, str]:
    """Pick the output frame rate and say where it came from.

    Order: an explicit ``override``; else the source video's FPS if valid;
    else :data:`FALLBACK_FPS` (the caller should warn that timing may be wrong).
    Returns ``(fps, source)`` with ``source`` in ``{"override", "source", "fallback"}``.
    """
    if override is not None:
        return validate_fps(override), "override"
    if source_fps is not None:
        try:
            return validate_fps(source_fps), "source"
        except ValueError:
            pass
    return FALLBACK_FPS, "fallback"


def partial_path_for(path: Path) -> Path:
    """``results/out.mp4`` -> ``results/out.partial.mp4`` (extension kept for OpenCV)."""
    return path.with_name(f"{path.stem}.partial{path.suffix}")


class VideoWriter:
    """Sequential, validated, all-or-nothing video file writer.

    Args:
        path: Output file. Its extension selects the container, e.g. ``.mp4``.
        fps: Frames per second, finite and > 0.
        frame_size: ``(width, height)`` in pixels; every frame must match it.
        codec: Four-character codec code (default ``"mp4v"``).
        overwrite: Allow replacing an existing file (only when ``finish()`` succeeds).

    Raises:
        ValueError: invalid path, fps, frame size or codec.
        VideoWriteError: odd width or height (OpenCV would silently crop the frames).
        FileExistsError: ``path`` exists and ``overwrite`` is False.
        MissingDependencyError: OpenCV is not installed.
        VideoWriteError: OpenCV could not open the output with this codec.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        fps: float,
        frame_size: tuple[int, int],
        codec: str = DEFAULT_CODEC,
        overwrite: bool = False,
    ) -> None:
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError(f"path must be a str or pathlib.Path, got {type(path).__name__}")
        self.path = Path(path)
        if not self.path.suffix:
            raise ValueError(f"output path needs a file extension such as .mp4, got {self.path}")
        if self.path.is_dir():
            raise ValueError(f"output path is a directory: {self.path}")
        self.fps = validate_fps(fps)
        if (
            not isinstance(frame_size, (tuple, list))
            or len(frame_size) != 2
            or not all(_is_integer(v) and v > 0 for v in frame_size)
        ):
            raise ValueError(
                f"frame_size must be (width, height) with positive integers, got {frame_size!r}"
            )
        self.width, self.height = (int(v) for v in frame_size)
        if self.width % 2 or self.height % 2:
            # Checked with opencv-python 4.14: both mp4v and MJPG output silently drop the
            # last row/column of odd-sized frames, so the video would not match the source.
            raise VideoWriteError(
                f"Cannot write {self.width} x {self.height} video: OpenCV's encoders need an even "
                "width and height and would silently crop the frames instead."
            )
        if not isinstance(codec, str) or len(codec) != 4:
            raise ValueError(f"codec must be a four-character code such as 'mp4v', got {codec!r}")
        self.codec = codec
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"{self.path} already exists; use --overwrite to replace it")

        self.partial_path = partial_path_for(self.path)
        self.frames_written = 0
        self._finished = False
        self._writer: Any = None

        cv2 = import_optional("cv2", display_name="OpenCV", extra="vision", purpose="write video")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(self.partial_path), cv2.VideoWriter_fourcc(*codec), self.fps, (self.width, self.height)
        )
        if not writer.isOpened():
            writer.release()
            self.partial_path.unlink(missing_ok=True)
            raise VideoWriteError(
                f"OpenCV could not open {self.path} for writing with codec {codec!r} "
                f"({self.width} x {self.height} at {self.fps:g} FPS). The codec may be unavailable "
                "in this OpenCV build, or the location may not be writable."
            )
        self._writer = writer

    def write(self, frame: np.ndarray) -> None:
        """Append one BGR frame. It must be exactly ``frame_size``."""
        if self._writer is None:
            raise VideoWriteError(f"The writer for {self.path} is already closed.")
        validate_bgr_frame(frame)
        height, width = frame.shape[:2]
        if (width, height) != (self.width, self.height):
            raise VideoWriteError(
                f"Frame {self.frames_written} is {width} x {height}, but {self.path} is being "
                f"written at {self.width} x {self.height}; all frames must have the same size."
            )
        self._writer.write(frame)
        self.frames_written += 1

    def finish(self) -> Path:
        """Close the file and give it its final name. Returns the final path."""
        if self._writer is None:
            raise VideoWriteError(f"The writer for {self.path} is already closed.")
        self._writer.release()
        self._writer = None
        if not self.partial_path.is_file():
            raise VideoWriteError(f"OpenCV did not create {self.partial_path}; nothing was written.")
        os.replace(self.partial_path, self.path)
        self._finished = True
        logger.info("Wrote %d frames to %s", self.frames_written, self.path)
        return self.path

    def abort(self) -> None:
        """Release the writer and delete the partial file."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if not self._finished:
            self.partial_path.unlink(missing_ok=True)

    @property
    def closed(self) -> bool:
        return self._writer is None

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if not self._finished:
            self.abort()
