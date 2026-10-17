"""Run a detector over a prerecorded video, frame by frame.

Data flow::

    VideoReader --frame, index--> Detector.detect --> FrameDetections --> summary / JSON

This module only *coordinates* existing components: ``VideoReader`` decodes
frames, any :class:`~edgevision.models.base.Detector` finds objects. It owns
no OpenCV or model code of its own.

Usage::

    from edgevision.models import PytorchYoloDetector
    from edgevision.pipeline import VideoDetectionPipeline

    with PytorchYoloDetector("models/yolov8n.pt") as detector:
        with VideoDetectionPipeline("data/input/sample.mp4", detector, max_frames=100) as pipeline:
            for result in pipeline:              # one FrameDetections per frame, in order
                print(result.frame_index, len(result.detections))
        print(pipeline.summary())

Design notes:

* Frames are processed one at a time and results are yielded immediately, so
  memory use does not grow with video length. The pipeline keeps only
  counters, never the frames or the detections themselves.
* The detector is created once by the caller and reused for every frame; the
  pipeline never loads or closes it.
* The video is released when iteration ends, when an error occurs, or when
  the ``with`` block exits.
* The summary is a description of what was processed, **not** a performance
  measurement: it contains no timing.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

from edgevision.models.base import Detection, Detector
from edgevision.video.reader import StopReason, VideoMetadata, VideoReader


class PipelineError(Exception):
    """Processing a video failed (for example, the detector raised on a frame)."""


@dataclass(frozen=True)
class FrameDetections:
    """The detections found in one video frame (possibly none)."""

    frame_index: int
    detections: tuple[Detection, ...]

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly form. ``frame_index`` is stored once, at frame level."""
        return {
            "frame_index": self.frame_index,
            "detections": [
                {k: v for k, v in d.to_dict().items() if k != "frame_index"} for d in self.detections
            ],
        }


@dataclass(frozen=True)
class PipelineSummary:
    """What was processed. Counts always satisfy
    ``frames_processed == frames_with_detections + frames_without_detections`` and
    ``total_detections == sum(detections_by_class.values())``."""

    video_path: Path
    metadata: VideoMetadata
    max_frames: int | None
    frames_processed: int
    total_detections: int
    frames_with_detections: int
    detections_by_class: dict[str, int]
    completed: bool  # True if iteration ran to the end (end of video or max_frames)
    stop_reason: StopReason | None  # None if not completed
    ended_before_reported_count: bool

    @property
    def frames_without_detections(self) -> int:
        return self.frames_processed - self.frames_with_detections

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames_processed": self.frames_processed,
            "frames_with_detections": self.frames_with_detections,
            "frames_without_detections": self.frames_without_detections,
            "total_detections": self.total_detections,
            "detections_by_class": dict(sorted(self.detections_by_class.items())),
            "max_frames": self.max_frames,
            "completed": self.completed,
            "stop_reason": None if self.stop_reason is None else self.stop_reason.value,
            "ended_before_reported_count": self.ended_before_reported_count,
        }


class VideoDetectionPipeline:
    """Feed every frame of a video to a detector, in order, one frame at a time.

    Args:
        video_path: Video file (``str`` or ``pathlib.Path``).
        detector: Any object implementing the ``Detector`` protocol, already
            constructed (the pipeline does not create or close it).
        max_frames: Process at most this many frames; ``None`` for the whole video.

    The video is opened immediately, so a missing or unreadable file fails
    here (``VideoError``), and ``metadata`` is available before processing.
    Iterate the pipeline once to process the video; call ``summary()``
    afterwards (or at any time, for the counts so far).
    """

    def __init__(
        self,
        video_path: str | os.PathLike[str],
        detector: Detector,
        *,
        max_frames: int | None = None,
    ) -> None:
        if not isinstance(detector, Detector):
            raise TypeError(
                f"detector must implement detect(frame, frame_index) and close(), "
                f"got {type(detector).__name__}"
            )
        self.detector = detector
        self._reader = VideoReader(video_path, max_frames=max_frames)
        self.video_path = self._reader.path
        self.metadata = self._reader.metadata
        self.max_frames = self._reader.max_frames

        self._started = False
        self._completed = False
        self._frames_processed = 0
        self._frames_with_detections = 0
        self._by_class: Counter[str] = Counter()

    # --- processing ------------------------------------------------------------

    def __iter__(self) -> Iterator[FrameDetections]:
        if self._started:
            raise PipelineError("This pipeline has already been run; create a new one.")
        self._started = True
        return self._run()

    def _run(self) -> Iterator[FrameDetections]:
        try:
            for frame_index, frame in self._reader:
                try:
                    detections = self.detector.detect(frame, frame_index=frame_index)
                except Exception as exc:
                    raise PipelineError(
                        f"Detection failed on frame {frame_index} of {self.video_path}: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                yield self._record(frame_index, detections)
            self._completed = True
        finally:
            self._reader.close()

    def _record(self, frame_index: int, detections: Any) -> FrameDetections:
        """Check the detector kept its side of the contract, then update the counters."""
        if not isinstance(detections, (list, tuple)) or not all(
            isinstance(d, Detection) for d in detections
        ):
            raise PipelineError(
                f"Detector returned {type(detections).__name__} for frame {frame_index}; "
                "expected a list of Detection records"
            )
        for d in detections:
            if d.frame_index != frame_index:
                raise PipelineError(
                    f"Detector labelled a detection from frame {frame_index} "
                    f"with frame_index={d.frame_index}"
                )
        self._frames_processed += 1
        if detections:
            self._frames_with_detections += 1
            self._by_class.update(d.class_name for d in detections)
        return FrameDetections(frame_index, tuple(detections))

    # --- results ----------------------------------------------------------------

    def summary(self) -> PipelineSummary:
        return PipelineSummary(
            video_path=self.video_path,
            metadata=self.metadata,
            max_frames=self.max_frames,
            frames_processed=self._frames_processed,
            total_detections=sum(self._by_class.values()),
            frames_with_detections=self._frames_with_detections,
            detections_by_class=dict(self._by_class),
            completed=self._completed,
            stop_reason=self._reader.stop_reason if self._completed else None,
            ended_before_reported_count=self._completed and self._reader.ended_before_reported_count,
        )

    # --- resource handling --------------------------------------------------------

    @property
    def closed(self) -> bool:
        return self._reader.closed

    def close(self) -> None:
        """Release the video. Does not close the detector, which belongs to the caller."""
        self._reader.close()

    def __enter__(self) -> VideoDetectionPipeline:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def process_video(
    video_path: str | os.PathLike[str],
    detector: Detector,
    *,
    max_frames: int | None = None,
) -> PipelineSummary:
    """Process a whole video and return only the summary (detections are not kept)."""
    with VideoDetectionPipeline(video_path, detector, max_frames=max_frames) as pipeline:
        for _ in pipeline:
            pass
    return pipeline.summary()
