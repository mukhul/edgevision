"""Video input, annotation and output. Importing this package does not require OpenCV."""

from edgevision.video.annotator import annotate_frame, class_color
from edgevision.video.reader import (
    Frame,
    MissingDependencyError,
    StopReason,
    VideoError,
    VideoMetadata,
    VideoReader,
)
from edgevision.video.writer import VideoWriteError, VideoWriter, choose_output_fps

__all__ = [
    "Frame",
    "MissingDependencyError",
    "StopReason",
    "VideoError",
    "VideoMetadata",
    "VideoReader",
    "VideoWriteError",
    "VideoWriter",
    "annotate_frame",
    "choose_output_fps",
    "class_color",
]
