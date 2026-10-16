"""Tests for edgevision.video.reader.

Unit tests replace OpenCV with a fake ``cv2`` module whose ``VideoCapture``
returns scripted frames and metadata. They therefore run without OpenCV,
without video codecs, and give the same result on every machine.

Integration tests at the end use the real OpenCV to write and read back a
tiny synthetic video. They are skipped (with the reason shown) only if
OpenCV is not installed or cannot encode video.
"""

from __future__ import annotations

import importlib.abc
import logging
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from edgevision.video import (
    Frame,
    MissingDependencyError,
    StopReason,
    VideoError,
    VideoMetadata,
    VideoReader,
)

# --- fake OpenCV -------------------------------------------------------------

# Property IDs are arbitrary here; the reader only ever uses them via cv2.CAP_PROP_*.
WIDTH, HEIGHT, FPS, FOURCC, COUNT = 3, 4, 5, 6, 7


def fourcc(code: str) -> float:
    """Pack a 4-character codec code the way OpenCV reports it."""
    return float(sum(ord(c) << (8 * i) for i, c in enumerate(code)))


GOOD_PROPS = {WIDTH: 640.0, HEIGHT: 480.0, FPS: 30.0, COUNT: 3.0, FOURCC: fourcc("mp4v")}


class FakeCapture:
    def __init__(self, path: str, frames: list[Any], props: dict[int, float], opened: bool) -> None:
        self.path = path
        self._frames = list(frames)
        self._props = props
        self._opened = opened
        self.read_calls = 0
        self.release_calls = 0

    def isOpened(self) -> bool:
        return self._opened

    def get(self, prop: int) -> float:
        return self._props.get(prop, 0.0)  # OpenCV returns 0 for unknown properties

    def read(self) -> tuple[bool, Any]:
        assert self.release_calls == 0, "read() called after release()"
        self.read_calls += 1
        if self._frames:
            return True, self._frames.pop(0)
        return False, None  # what OpenCV returns at the end of a video

    def release(self) -> None:
        self.release_calls += 1

    @property
    def released(self) -> bool:
        return self.release_calls > 0


def fake_image(index: int) -> SimpleNamespace:
    return SimpleNamespace(shape=(480, 640, 3), marker=index)


@pytest.fixture
def video_file(tmp_path: Path) -> Path:
    """A file that exists. Its content is irrelevant because cv2 is faked."""
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"not decoded by the fake")
    return path


@pytest.fixture
def fake_cv2(monkeypatch: pytest.MonkeyPatch):
    """Install a fake cv2 module. Call it to configure what the video contains."""
    captures: list[FakeCapture] = []

    def install(
        frames: list[Any] | None = None,
        props: dict[int, float] | None = None,
        opened: bool = True,
    ) -> list[FakeCapture]:
        frames = [fake_image(i) for i in range(3)] if frames is None else frames
        props = GOOD_PROPS if props is None else props

        def video_capture(path: str) -> FakeCapture:
            capture = FakeCapture(path, frames, props, opened)
            captures.append(capture)
            return capture

        module = SimpleNamespace(
            VideoCapture=video_capture,
            CAP_PROP_FRAME_WIDTH=WIDTH,
            CAP_PROP_FRAME_HEIGHT=HEIGHT,
            CAP_PROP_FPS=FPS,
            CAP_PROP_FOURCC=FOURCC,
            CAP_PROP_FRAME_COUNT=COUNT,
        )
        monkeypatch.setitem(sys.modules, "cv2", module)
        return captures

    return install


# --- input validation --------------------------------------------------------


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(VideoError, match="Video file not found") as info:
        VideoReader(tmp_path / "missing.mp4")
    assert "missing.mp4" in str(info.value)


def test_directory_instead_of_file(tmp_path: Path) -> None:
    with pytest.raises(VideoError, match="not a file"):
        VideoReader(tmp_path)


def test_path_of_wrong_type() -> None:
    with pytest.raises(TypeError, match="str or pathlib.Path"):
        VideoReader(42)  # type: ignore[arg-type]


def test_str_and_path_are_both_accepted(fake_cv2, video_file: Path) -> None:
    captures = fake_cv2()
    VideoReader(str(video_file)).close()
    VideoReader(video_file).close()
    assert [c.path for c in captures] == [str(video_file)] * 2


@pytest.mark.parametrize(
    ("max_frames", "error"),
    [(0, ValueError), (-5, ValueError), (True, TypeError), (False, TypeError),
     (2.5, TypeError), ("10", TypeError)],
)
def test_invalid_max_frames_is_rejected(
    fake_cv2, video_file: Path, max_frames: Any, error: type[Exception]
) -> None:
    captures = fake_cv2()
    with pytest.raises(error, match="max_frames"):
        VideoReader(video_file, max_frames=max_frames)
    assert captures == []  # rejected before any video was opened


def test_video_that_cannot_be_opened(fake_cv2, video_file: Path) -> None:
    captures = fake_cv2(opened=False)
    with pytest.raises(VideoError, match="could not open"):
        VideoReader(video_file)
    assert captures[0].released


# --- metadata ----------------------------------------------------------------


def test_metadata_is_read(fake_cv2, video_file: Path) -> None:
    fake_cv2()
    with VideoReader(video_file) as reader:
        assert reader.metadata == VideoMetadata(
            path=video_file, width=640, height=480, fps=30.0, frame_count=3, fourcc="mp4v"
        )
        assert reader.metadata.duration_s == pytest.approx(0.1)


@pytest.mark.parametrize(
    "props",
    [
        {},  # nothing reported: OpenCV returns 0.0 for everything
        {WIDTH: -1.0, HEIGHT: 0.0, FPS: math.nan, COUNT: -1.0, FOURCC: 0.0},
        {WIDTH: math.inf, HEIGHT: math.nan, FPS: math.inf, COUNT: math.inf, FOURCC: -1.0},
    ],
)
def test_unknown_or_invalid_metadata_becomes_none(fake_cv2, video_file: Path, props: dict) -> None:
    fake_cv2(props=props)
    with VideoReader(video_file) as reader:
        m = reader.metadata
        assert (m.width, m.height, m.fps, m.frame_count, m.fourcc) == (None,) * 5
        assert m.duration_s is None
        assert len(list(reader)) == 3  # reading still works


def test_fractional_frame_count_is_rounded(fake_cv2, video_file: Path) -> None:
    fake_cv2(props={**GOOD_PROPS, COUNT: 299.9999})
    with VideoReader(video_file) as reader:
        assert reader.metadata.frame_count == 300


# --- reading frames ----------------------------------------------------------


def test_frames_are_yielded_in_order_with_zero_based_indices(fake_cv2, video_file: Path) -> None:
    images = [fake_image(i) for i in range(3)]
    fake_cv2(frames=images)

    with VideoReader(video_file) as reader:
        frames = list(reader)

    assert [index for index, _ in frames] == [0, 1, 2]
    # The exact objects OpenCV returned are passed through: no copy, no BGR->RGB conversion.
    assert all(frame.image is image for frame, image in zip(frames, images))
    assert all(isinstance(frame, Frame) for frame in frames)


def test_end_of_video_stops_iteration_cleanly(fake_cv2, video_file: Path) -> None:
    captures = fake_cv2()
    with VideoReader(video_file, max_frames=10) as reader:
        assert len(list(reader)) == 3
    assert reader.frames_read == 3
    assert reader.stop_reason is StopReason.END_OF_VIDEO
    assert not reader.ended_before_reported_count
    assert captures[0].read_calls == 4  # three frames, then the end-of-video signal


def test_max_frames_stops_at_the_requested_count(fake_cv2, video_file: Path) -> None:
    captures = fake_cv2(frames=[fake_image(i) for i in range(10)])
    with VideoReader(video_file, max_frames=4) as reader:
        indices = [index for index, _ in reader]
    assert indices == [0, 1, 2, 3]
    assert reader.stop_reason is StopReason.MAX_FRAMES
    assert captures[0].read_calls == 4  # it does not read ahead past the limit


def test_max_frames_equal_to_video_length(fake_cv2, video_file: Path) -> None:
    fake_cv2()
    with VideoReader(video_file, max_frames=3) as reader:
        assert len(list(reader)) == 3
    assert reader.stop_reason is StopReason.MAX_FRAMES


def test_video_with_no_decodable_frames(fake_cv2, video_file: Path) -> None:
    captures = fake_cv2(frames=[])
    reader = VideoReader(video_file)
    with pytest.raises(VideoError, match="could not decode any frames"):
        list(reader)
    assert captures[0].released


def test_video_ending_before_reported_count_is_flagged(
    fake_cv2, video_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    fake_cv2(props={**GOOD_PROPS, COUNT: 10.0})
    with caplog.at_level(logging.WARNING), VideoReader(video_file) as reader:
        list(reader)
    assert reader.ended_before_reported_count
    assert "ended after 3 frames, but the file reported 10" in caplog.text


def test_reader_is_single_use(fake_cv2, video_file: Path) -> None:
    fake_cv2()
    with VideoReader(video_file) as reader:
        list(reader)
        with pytest.raises(VideoError, match="already been read"):
            iter(reader)


# --- resource cleanup --------------------------------------------------------


def test_released_after_normal_iteration_without_with(fake_cv2, video_file: Path) -> None:
    captures = fake_cv2()
    reader = VideoReader(video_file)
    list(reader)
    assert captures[0].released
    assert reader.closed


def test_released_when_with_block_exits_normally(fake_cv2, video_file: Path) -> None:
    captures = fake_cv2()
    with VideoReader(video_file) as reader:
        frames = iter(reader)
        next(frames)  # read only part of the video, keeping the iterator alive
        assert not captures[0].released
    assert captures[0].released  # released by the with block, not by the iterator
    assert captures[0].release_calls == 1


def test_released_when_an_exception_occurs_inside_with(fake_cv2, video_file: Path) -> None:
    captures = fake_cv2()
    with pytest.raises(RuntimeError, match="boom"):
        with VideoReader(video_file) as reader:
            for _ in reader:
                raise RuntimeError("boom")
    assert captures[0].released
    assert captures[0].release_calls == 1


def test_released_when_iteration_is_abandoned(fake_cv2, video_file: Path) -> None:
    captures = fake_cv2()
    frames = iter(VideoReader(video_file))
    next(frames)
    frames.close()  # what Python does when a for-loop over the reader is exited early
    assert captures[0].released


def test_close_is_idempotent_and_metadata_survives(fake_cv2, video_file: Path) -> None:
    captures = fake_cv2()
    reader = VideoReader(video_file)
    reader.close()
    reader.close()
    assert captures[0].release_calls == 1
    assert reader.metadata.width == 640
    with pytest.raises(VideoError, match="closed"):
        iter(reader)


# --- missing or broken OpenCV ------------------------------------------------


def test_missing_opencv_gives_install_instructions(
    monkeypatch: pytest.MonkeyPatch, video_file: Path
) -> None:
    monkeypatch.setitem(sys.modules, "cv2", None)  # behaves exactly like "not installed"
    with pytest.raises(MissingDependencyError) as info:
        VideoReader(video_file)
    message = str(info.value)
    assert "OpenCV is required" in message
    assert 'pip install -e ".[vision]"' in message
    assert isinstance(info.value, ImportError)


def test_broken_opencv_install_is_reported(monkeypatch: pytest.MonkeyPatch, video_file: Path) -> None:
    class BrokenCv2Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name: str, path: Any, target: Any = None) -> None:
            if name == "cv2":
                raise ImportError("libGL.so.1: cannot open shared object file", name="libGL")
            return None

    monkeypatch.delitem(sys.modules, "cv2", raising=False)
    monkeypatch.setattr(sys, "meta_path", [BrokenCv2Finder(), *sys.meta_path])
    with pytest.raises(MissingDependencyError, match="installed but could not be imported.*libGL"):
        VideoReader(video_file)


def test_package_imports_without_opencv() -> None:
    # A fresh interpreter in which cv2 cannot be imported at all.
    code = (
        "import sys; sys.modules['cv2'] = None; "
        "import edgevision, edgevision.video, edgevision.config, edgevision.device; print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


# --- integration: real OpenCV, synthetic video ---------------------------------

SYNTHETIC_SIZE = (64, 48)  # width, height
SYNTHETIC_FRAMES = 12
SYNTHETIC_FPS = 10.0


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Write a tiny Motion-JPEG AVI whose frames are pure blue in BGR order."""
    cv2 = pytest.importorskip("cv2", reason='OpenCV not installed (pip install -e ".[vision]")')
    np = pytest.importorskip("numpy")

    path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), SYNTHETIC_FPS, SYNTHETIC_SIZE)
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot encode MJPG video")
    width, height = SYNTHETIC_SIZE
    try:
        for _ in range(SYNTHETIC_FRAMES):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = 255  # channel 0 is BLUE in OpenCV's BGR order
            writer.write(frame)
    finally:
        writer.release()
    return path


def test_integration_reads_synthetic_video(synthetic_video: Path) -> None:
    with VideoReader(synthetic_video) as reader:
        metadata = reader.metadata
        frames = list(reader)

    assert (metadata.width, metadata.height) == SYNTHETIC_SIZE
    assert metadata.fps == pytest.approx(SYNTHETIC_FPS)
    assert metadata.frame_count == SYNTHETIC_FRAMES
    assert metadata.fourcc == "MJPG"

    assert [index for index, _ in frames] == list(range(SYNTHETIC_FRAMES))
    assert reader.stop_reason is StopReason.END_OF_VIDEO
    image = frames[0].image
    assert image.shape == (48, 64, 3)
    assert str(image.dtype) == "uint8"
    # JPEG is lossy, so compare channel averages rather than exact values.
    blue, green, red = (float(image[:, :, c].mean()) for c in range(3))
    assert blue > 200 and green < 50 and red < 50  # still blue: channel order is BGR


def test_integration_max_frames(synthetic_video: Path) -> None:
    with VideoReader(synthetic_video, max_frames=5) as reader:
        assert len(list(reader)) == 5
    assert reader.stop_reason is StopReason.MAX_FRAMES


def test_integration_rejects_file_that_is_not_a_video(tmp_path: Path) -> None:
    pytest.importorskip("cv2", reason='OpenCV not installed (pip install -e ".[vision]")')
    path = tmp_path / "notes.mp4"
    path.write_text("this is text, not video", encoding="utf-8")
    # Depending on the OpenCV backend this fails either when opening or on the first read.
    with pytest.raises(VideoError):
        with VideoReader(path) as reader:
            list(reader)
