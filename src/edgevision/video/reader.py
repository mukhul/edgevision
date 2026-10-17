"""Read prerecorded video files frame by frame with OpenCV.

Usage::

    from edgevision.video import VideoReader

    with VideoReader("data/input/sample.mp4", max_frames=100) as reader:
        print(reader.metadata)
        for frame_index, frame in reader:
            ...  # frame: BGR uint8 NumPy array, shape (height, width, 3)

    print(reader.frames_read, reader.stop_reason)

Design notes:

* Frames are decoded one at a time, so memory use does not grow with video
  length.
* Frames are passed through exactly as OpenCV decodes them (BGR channel
  order, no resizing). Model-specific preprocessing belongs elsewhere.
* A reader is single-use: iterate it once, then create a new one to read
  the video again.
* OpenCV is optional. It is imported only when a reader is created, so the
  rest of the package works without it; install it with
  ``python -m pip install -e ".[vision]"``.
"""

from __future__ import annotations

import enum
import logging
import math
import numbers
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:  # only for type checkers; NumPy is not needed to import this module
    import numpy as np

from edgevision.optional import MissingDependencyError, import_optional

logger = logging.getLogger(__name__)

__all__ = [
    "Frame",
    "MissingDependencyError",  # re-exported: defined in edgevision.optional
    "StopReason",
    "VideoError",
    "VideoMetadata",
    "VideoReader",
]


class VideoError(Exception):
    """A video file could not be found, opened, or decoded."""


class StopReason(enum.StrEnum):
    """Why iteration over a video finished."""

    END_OF_VIDEO = "end_of_video"  # OpenCV returned no further frame
    MAX_FRAMES = "max_frames"  # the max_frames limit was reached


@dataclass(frozen=True)
class VideoMetadata:
    """Properties *reported by the file*. Any value is ``None`` if it is missing or invalid.

    Reported values come from the container header and can be wrong: the
    frame count in particular is often an estimate. The number of frames
    actually decoded is ``VideoReader.frames_read``.
    """

    path: Path
    width: int | None
    height: int | None
    fps: float | None
    frame_count: int | None
    fourcc: str | None  # four-character codec code, e.g. "avc1", "mp4v", "MJPG"

    @property
    def duration_s(self) -> float | None:
        """Reported duration in seconds (frame_count / fps), if both are known."""
        if self.fps is None or self.frame_count is None:
            return None
        return self.frame_count / self.fps


class Frame(NamedTuple):
    """One decoded frame. Unpacks as ``index, image``."""

    index: int  # zero-based position in the sequence of decoded frames
    image: np.ndarray  # BGR, uint8, shape (height, width, 3)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_cv2() -> Any:
    """Import OpenCV, turning a missing or broken install into an actionable error."""
    return import_optional("cv2", display_name="OpenCV", extra="vision", purpose="read video")


def _validate_path(path: Any) -> Path:
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(f"video path must be a str or pathlib.Path, got {type(path).__name__}")
    path = Path(path)
    if not path.exists():
        raise VideoError(f"Video file not found: {path} (looked relative to {Path.cwd()})")
    if not path.is_file():
        raise VideoError(f"Video path is not a file: {path}")
    return path


def _validate_max_frames(max_frames: Any) -> int | None:
    if max_frames is None:
        return None
    # bool is a subclass of int, so True would otherwise be accepted as 1.
    if isinstance(max_frames, bool) or not isinstance(max_frames, numbers.Integral):
        raise TypeError(
            f"max_frames must be a positive integer or None, got {max_frames!r} "
            f"({type(max_frames).__name__})"
        )
    if max_frames < 1:
        raise ValueError(
            f"max_frames must be at least 1 (use None to read the whole video), got {max_frames}"
        )
    return int(max_frames)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive_int(value: Any) -> int | None:
    number = _finite(value)
    return round(number) if number is not None and number >= 1 else None


def _positive_float(value: Any) -> float | None:
    number = _finite(value)
    return number if number is not None and number > 0 else None


def _fourcc_text(value: Any) -> str | None:
    number = _finite(value)
    if number is None or number <= 0:
        return None
    code = int(number)
    text = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
    return text if text.isprintable() and text.strip() else None


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class VideoReader:
    """Sequential, single-pass reader for a prerecorded video file.

    Args:
        path: Video file to read (``str`` or ``pathlib.Path``).
        max_frames: Stop after this many frames. ``None`` reads the whole video.

    Raises:
        TypeError / ValueError: if ``path`` or ``max_frames`` has the wrong type or value.
        VideoError: if the file does not exist, is not a file, or OpenCV cannot open it.
        MissingDependencyError: if OpenCV is not installed.

    After iteration, ``frames_read`` holds the number of frames delivered and
    ``stop_reason`` says why reading ended. The underlying ``cv2.VideoCapture``
    is released when iteration ends, when ``close()`` is called, or when a
    ``with`` block exits, including when an exception occurs.
    """

    def __init__(self, path: str | os.PathLike[str], max_frames: int | None = None) -> None:
        # Check the arguments first, so mistakes are reported even without OpenCV.
        self.max_frames = _validate_max_frames(max_frames)
        self.path = _validate_path(path)
        self.frames_read = 0
        self.stop_reason: StopReason | None = None
        self._started = False

        cv2 = _load_cv2()
        self._capture: Any = cv2.VideoCapture(str(self.path))
        try:
            if not self._capture.isOpened():
                raise VideoError(
                    f"OpenCV could not open {self.path}. The file may be corrupt, not a video, "
                    "or use a codec this OpenCV build cannot decode."
                )
            self.metadata = self._read_metadata(cv2)
        except BaseException:
            self.close()
            raise
        logger.debug("Opened %s: %s", self.path, self.metadata)

    def _read_metadata(self, cv2: Any) -> VideoMetadata:
        get = self._capture.get
        return VideoMetadata(
            path=self.path,
            width=_positive_int(get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=_positive_int(get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=_positive_float(get(cv2.CAP_PROP_FPS)),
            frame_count=_positive_int(get(cv2.CAP_PROP_FRAME_COUNT)),
            fourcc=_fourcc_text(get(cv2.CAP_PROP_FOURCC)),
        )

    # --- iteration ---------------------------------------------------------

    def __iter__(self) -> Iterator[Frame]:
        # These checks run immediately, not when the first frame is requested.
        if self._started:
            raise VideoError(
                f"{self.path} has already been read by this VideoReader; "
                "create a new VideoReader to read it again."
            )
        if self.closed:
            raise VideoError(f"This VideoReader for {self.path} is closed.")
        self._started = True
        return self._frames()

    def _frames(self) -> Iterator[Frame]:
        try:
            while self.max_frames is None or self.frames_read < self.max_frames:
                ok, image = self._capture.read()
                if not ok or image is None:
                    # OpenCV reports the end of the video and an undecodable
                    # frame the same way, so both end iteration here.
                    self.stop_reason = StopReason.END_OF_VIDEO
                    self._check_ending()
                    return
                index = self.frames_read
                self.frames_read += 1
                yield Frame(index, image)
            self.stop_reason = StopReason.MAX_FRAMES
        finally:
            self.close()

    def _check_ending(self) -> None:
        if self.frames_read == 0:
            raise VideoError(
                f"OpenCV opened {self.path} but could not decode any frames. The file may be "
                "corrupt or use a codec this OpenCV build cannot decode."
            )
        if self.ended_before_reported_count:
            logger.warning(
                "%s ended after %d frames, but the file reported %d. It may be truncated, "
                "or its reported frame count may be inaccurate.",
                self.path, self.frames_read, self.metadata.frame_count,
            )

    @property
    def ended_before_reported_count(self) -> bool:
        """True if the video ended before the frame count reported by the file."""
        reported = self.metadata.frame_count
        return (
            self.stop_reason is StopReason.END_OF_VIDEO
            and reported is not None
            and self.frames_read < reported
        )

    # --- resource handling ---------------------------------------------------

    @property
    def closed(self) -> bool:
        return self._capture is None

    def close(self) -> None:
        """Release the OpenCV capture. Safe to call more than once."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()  # returning None lets any exception continue to propagate

    def __repr__(self) -> str:
        state = "closed" if self.closed else "open"
        return f"VideoReader({str(self.path)!r}, max_frames={self.max_frames}, {state})"
