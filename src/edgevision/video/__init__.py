"""Video input. Importing this package does not require OpenCV."""

from edgevision.video.reader import (
    Frame,
    MissingDependencyError,
    StopReason,
    VideoError,
    VideoMetadata,
    VideoReader,
)

__all__ = [
    "Frame",
    "MissingDependencyError",
    "StopReason",
    "VideoError",
    "VideoMetadata",
    "VideoReader",
]
