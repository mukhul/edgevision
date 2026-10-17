"""Tests for edgevision.pipeline.

Most tests replace the video reader and the detector with small fakes, so the
pipeline's own logic (order, frame indices, counting, error handling,
cleanup) is tested deterministically, without OpenCV, codecs or YOLO.
Integration tests use the real VideoReader on a synthetic video (needs
OpenCV), and one opt-in test runs the real YOLOv8n model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import edgevision.pipeline as pipeline_module
from edgevision.models.base import Detection
from edgevision.pipeline import (
    FrameDetections,
    PipelineError,
    VideoDetectionPipeline,
    process_video,
)
from edgevision.video.reader import Frame, StopReason, VideoError, VideoMetadata

REPO = Path(__file__).resolve().parents[1]


# --- fakes -------------------------------------------------------------------------


@dataclass
class FakeVideo:
    """Controls what the fake reader 'decodes'."""

    frames: int = 5
    reported_count: int | None = None
    readers: list[FakeReader] = field(default_factory=list)


class FakeReader:
    """Stands in for VideoReader: same attributes, yields placeholder frames."""

    def __init__(self, video: FakeVideo, path: Any, max_frames: int | None) -> None:
        self.path = Path(path)
        self.max_frames = max_frames
        self.metadata = VideoMetadata(self.path, 64, 48, 30.0, video.reported_count or video.frames, "MJPG")
        self.frames_read = 0
        self.stop_reason: StopReason | None = None
        self.ended_before_reported_count = False
        self.release_calls = 0
        self._video = video
        self._closed = False

    def __iter__(self):
        try:
            while self.max_frames is None or self.frames_read < self.max_frames:
                if self.frames_read >= self._video.frames:
                    self.stop_reason = StopReason.END_OF_VIDEO
                    reported = self.metadata.frame_count
                    self.ended_before_reported_count = reported is not None and self.frames_read < reported
                    return
                index = self.frames_read
                self.frames_read += 1
                yield Frame(index, f"image-{index}")
            self.stop_reason = StopReason.MAX_FRAMES
        finally:
            self.close()

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.release_calls += 1


@pytest.fixture
def fake_video(monkeypatch: pytest.MonkeyPatch) -> FakeVideo:
    video = FakeVideo()

    def make_reader(path: Any, max_frames: int | None = None) -> FakeReader:
        reader = FakeReader(video, path, max_frames)
        video.readers.append(reader)
        return reader

    monkeypatch.setattr(pipeline_module, "VideoReader", make_reader)
    return video


class FakeDetector:
    """Returns scripted detections per frame index and records every call."""

    def __init__(self, per_frame: dict[int, list[str]] | None = None, fail_at: int | None = None) -> None:
        self.per_frame = per_frame or {}
        self.fail_at = fail_at
        self.calls: list[tuple[Any, int]] = []
        self.closed = False

    def detect(self, frame: Any, frame_index: int = 0) -> list[Detection]:
        self.calls.append((frame, frame_index))
        if frame_index == self.fail_at:
            raise RuntimeError("CUDA out of memory")
        return [
            Detection(frame_index, (1.0, 2.0, 30.0, 40.0), 0.8, 0, name)
            for name in self.per_frame.get(frame_index, [])
        ]

    def close(self) -> None:
        self.closed = True


def run_all(pipeline: VideoDetectionPipeline) -> list[FrameDetections]:
    with pipeline:
        return list(pipeline)


# --- order, frame indices and aggregation --------------------------------------------


def test_frames_are_processed_in_order_with_correct_indices(fake_video: FakeVideo) -> None:
    detector = FakeDetector()
    results = run_all(VideoDetectionPipeline("clip.mp4", detector))

    assert detector.calls == [(f"image-{i}", i) for i in range(5)]  # each frame, its own index
    assert [r.frame_index for r in results] == [0, 1, 2, 3, 4]


def test_detections_are_aggregated_correctly(fake_video: FakeVideo) -> None:
    detector = FakeDetector({0: ["cat"], 2: ["cat", "dog", "person"], 4: ["dog"]})
    pipeline = VideoDetectionPipeline("clip.mp4", detector)
    results = run_all(pipeline)
    summary = pipeline.summary()

    assert [len(r.detections) for r in results] == [1, 0, 3, 0, 1]
    assert all(d.frame_index == r.frame_index for r in results for d in r.detections)
    assert summary.frames_processed == 5
    assert summary.total_detections == 5
    assert summary.frames_with_detections == 3
    assert summary.frames_without_detections == 2
    assert summary.detections_by_class == {"cat": 2, "dog": 2, "person": 1}
    assert summary.completed and summary.stop_reason is StopReason.END_OF_VIDEO


def test_video_without_any_detections(fake_video: FakeVideo) -> None:
    pipeline = VideoDetectionPipeline("clip.mp4", FakeDetector())
    results = run_all(pipeline)
    summary = pipeline.summary()

    assert all(r.detections == () for r in results)
    assert (summary.frames_processed, summary.total_detections) == (5, 0)
    assert (summary.frames_with_detections, summary.frames_without_detections) == (0, 5)
    assert summary.detections_by_class == {}


def test_empty_video_gives_a_consistent_summary(fake_video: FakeVideo) -> None:
    fake_video.frames = 0
    pipeline = VideoDetectionPipeline("clip.mp4", FakeDetector())
    assert run_all(pipeline) == []
    s = pipeline.summary()
    assert (s.frames_processed, s.total_detections, s.frames_with_detections, s.frames_without_detections) == (0, 0, 0, 0)
    assert s.completed


def test_summary_before_processing_is_all_zero_and_not_completed(fake_video: FakeVideo) -> None:
    with VideoDetectionPipeline("clip.mp4", FakeDetector()) as pipeline:
        s = pipeline.summary()
    assert (s.frames_processed, s.total_detections, s.frames_without_detections) == (0, 0, 0)
    assert not s.completed and s.stop_reason is None


def test_max_frames_is_respected(fake_video: FakeVideo) -> None:
    fake_video.frames = 50
    detector = FakeDetector()
    pipeline = VideoDetectionPipeline("clip.mp4", detector, max_frames=3)
    results = run_all(pipeline)

    assert [r.frame_index for r in results] == [0, 1, 2]
    assert len(detector.calls) == 3  # no extra frames were sent to the detector
    assert pipeline.summary().stop_reason is StopReason.MAX_FRAMES
    assert pipeline.summary().max_frames == 3


def test_summary_reports_video_ending_before_reported_count(fake_video: FakeVideo) -> None:
    fake_video.frames, fake_video.reported_count = 4, 10
    pipeline = VideoDetectionPipeline("clip.mp4", FakeDetector())
    run_all(pipeline)
    assert pipeline.summary().ended_before_reported_count


def test_results_are_streamed_one_frame_at_a_time(fake_video: FakeVideo) -> None:
    detector = FakeDetector()
    with VideoDetectionPipeline("clip.mp4", detector) as pipeline:
        frames = iter(pipeline)
        first = next(frames)
        assert first.frame_index == 0
        assert len(detector.calls) == 1  # nothing beyond the first frame has been processed yet


def test_process_video_returns_the_summary(fake_video: FakeVideo) -> None:
    summary = process_video("clip.mp4", FakeDetector({1: ["cat"]}), max_frames=2)
    assert (summary.frames_processed, summary.total_detections) == (2, 1)


def test_summary_to_dict_is_consistent(fake_video: FakeVideo) -> None:
    pipeline = VideoDetectionPipeline("clip.mp4", FakeDetector({0: ["dog", "cat"]}))
    run_all(pipeline)
    d = pipeline.summary().to_dict()
    assert d["frames_with_detections"] + d["frames_without_detections"] == d["frames_processed"]
    assert sum(d["detections_by_class"].values()) == d["total_detections"]
    assert list(d["detections_by_class"]) == ["cat", "dog"]  # sorted for stable output
    assert d["stop_reason"] == "end_of_video"


# --- errors -------------------------------------------------------------------------


def test_detector_failure_is_propagated_with_context(fake_video: FakeVideo) -> None:
    detector = FakeDetector(fail_at=2)
    pipeline = VideoDetectionPipeline("clip.mp4", detector)
    with pytest.raises(PipelineError) as info:
        run_all(pipeline)

    message = str(info.value)
    assert "frame 2" in message and "clip.mp4" in message and "CUDA out of memory" in message
    assert isinstance(info.value.__cause__, RuntimeError)  # original error kept
    assert pipeline.summary().frames_processed == 2  # frames 0 and 1 were finished
    assert not pipeline.summary().completed


def test_detector_returning_wrong_frame_index_is_caught(fake_video: FakeVideo) -> None:
    class MislabellingDetector(FakeDetector):
        def detect(self, frame: Any, frame_index: int = 0) -> list[Detection]:
            return [Detection(frame_index + 1, (0, 0, 1, 1), 0.5, 0, "cat")]

    with pytest.raises(PipelineError, match="frame_index=1"):
        run_all(VideoDetectionPipeline("clip.mp4", MislabellingDetector()))


def test_detector_returning_wrong_type_is_caught(fake_video: FakeVideo) -> None:
    class BadDetector(FakeDetector):
        def detect(self, frame: Any, frame_index: int = 0) -> Any:
            return {"boxes": []}

    with pytest.raises(PipelineError, match="expected a list of Detection"):
        run_all(VideoDetectionPipeline("clip.mp4", BadDetector()))


def test_object_without_detect_method_is_rejected(fake_video: FakeVideo) -> None:
    with pytest.raises(TypeError, match="detect"):
        VideoDetectionPipeline("clip.mp4", object())  # type: ignore[arg-type]


def test_missing_video_gives_a_clear_error(tmp_path: Path) -> None:
    # Real VideoReader: path checks happen before OpenCV is needed.
    with pytest.raises(VideoError, match="Video file not found"):
        VideoDetectionPipeline(tmp_path / "missing.mp4", FakeDetector())


def test_pipeline_runs_only_once(fake_video: FakeVideo) -> None:
    with VideoDetectionPipeline("clip.mp4", FakeDetector()) as pipeline:
        list(pipeline)
        with pytest.raises(PipelineError, match="already been run"):
            iter(pipeline)


# --- resource cleanup ---------------------------------------------------------------


def test_video_is_released_after_success(fake_video: FakeVideo) -> None:
    pipeline = VideoDetectionPipeline("clip.mp4", FakeDetector())
    list(pipeline)  # no with-block: iteration itself releases the video
    assert fake_video.readers[0].closed
    assert pipeline.closed


def test_video_is_released_when_inference_fails(fake_video: FakeVideo) -> None:
    pipeline = VideoDetectionPipeline("clip.mp4", FakeDetector(fail_at=0))
    with pytest.raises(PipelineError):
        list(pipeline)
    assert fake_video.readers[0].closed


def test_video_is_released_when_caller_stops_early(fake_video: FakeVideo) -> None:
    with VideoDetectionPipeline("clip.mp4", FakeDetector()) as pipeline:
        for result in pipeline:
            if result.frame_index == 1:
                break
    assert fake_video.readers[0].closed
    assert not pipeline.summary().completed


def test_pipeline_does_not_close_the_callers_detector(fake_video: FakeVideo) -> None:
    detector = FakeDetector()
    run_all(VideoDetectionPipeline("clip.mp4", detector))
    run_all(VideoDetectionPipeline("clip.mp4", detector))  # reused, not reloaded
    assert not detector.closed
    assert len(detector.calls) == 10


# --- integration: real VideoReader on a synthetic video (needs OpenCV) ---------------


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    cv2 = pytest.importorskip("cv2", reason='OpenCV not installed (pip install -e ".[vision]")')
    np = pytest.importorskip("numpy")
    path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot encode MJPG video")
    try:
        for _ in range(8):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    finally:
        writer.release()
    return path


def test_integration_real_reader_feeds_real_frames(synthetic_video: Path) -> None:
    detector = FakeDetector({3: ["cat"]})
    pipeline = VideoDetectionPipeline(synthetic_video, detector, max_frames=5)
    results = run_all(pipeline)

    assert [r.frame_index for r in results] == [0, 1, 2, 3, 4]
    assert all(frame.shape == (48, 64, 3) for frame, _ in detector.calls)  # real BGR arrays
    assert pipeline.summary().total_detections == 1
    assert pipeline.closed


def test_integration_real_reader_released_on_inference_failure(synthetic_video: Path) -> None:
    pipeline = VideoDetectionPipeline(synthetic_video, FakeDetector(fail_at=1))
    with pytest.raises(PipelineError):
        run_all(pipeline)
    assert pipeline.closed


def test_integration_file_with_no_decodable_frames_is_an_error(tmp_path: Path) -> None:
    pytest.importorskip("cv2", reason='OpenCV not installed (pip install -e ".[vision]")')
    path = tmp_path / "broken.mp4"
    path.write_text("not a video", encoding="utf-8")
    with pytest.raises(VideoError):  # raised when opening or on the first read
        run_all(VideoDetectionPipeline(path, FakeDetector()))


# --- real model (opt-in) ---------------------------------------------------------------

RUN_MODEL_TESTS = os.environ.get("EDGEVISION_RUN_MODEL_TESTS") == "1"
WEIGHTS = Path(os.environ.get("EDGEVISION_MODEL_WEIGHTS", REPO / "models" / "yolov8n.pt"))


@pytest.mark.model
@pytest.mark.skipif(
    not RUN_MODEL_TESTS,
    reason="real-model test is opt-in: set EDGEVISION_RUN_MODEL_TESTS=1 (needs the detector extra and models/yolov8n.pt)",
)
def test_real_yolov8n_on_a_short_video(tmp_path: Path) -> None:
    """Builds a 6-frame video from Ultralytics' bundled bus.jpg and runs the real model on it."""
    if not WEIGHTS.is_file():
        pytest.skip(f"weights not found at {WEIGHTS}; run scripts/detect_image.py once to download them")
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    pytest.importorskip("ultralytics")
    from ultralytics.utils import ASSETS

    from edgevision.models import PytorchYoloDetector

    image = cv2.resize(cv2.imread(str(ASSETS / "bus.jpg")), (480, 640))  # (width, height)
    video = tmp_path / "bus.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (480, 640))
    for shift in range(6):  # move the scene a few pixels right each frame
        matrix = np.float32([[1, 0, 4 * shift], [0, 1, 0]])
        writer.write(cv2.warpAffine(image, matrix, (480, 640)))
    writer.release()

    with PytorchYoloDetector(WEIGHTS, device="cpu") as detector:
        pipeline = VideoDetectionPipeline(video, detector)
        results = run_all(pipeline)

    summary = pipeline.summary()
    assert summary.frames_processed == 6 and summary.completed
    assert [r.frame_index for r in results] == list(range(6))
    assert all(any(d.class_name == "bus" for d in r.detections) for r in results)
    assert summary.total_detections == sum(len(r.detections) for r in results)
