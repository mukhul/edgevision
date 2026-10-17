"""Tests for edgevision.video.writer.

Unit tests replace cv2.VideoWriter with a fake that records what it is given,
so they are deterministic and independent of codecs. Integration tests write
real MP4 files with OpenCV and read them back with VideoReader.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from edgevision.optional import MissingDependencyError
from edgevision.video.writer import (
    FALLBACK_FPS,
    VideoWriteError,
    VideoWriter,
    choose_output_fps,
    partial_path_for,
)

SIZE = (64, 48)  # width, height


def frame(width: int = 64, height: int = 48, value: int = 0) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


class FakeCvWriter:
    def __init__(self, path: str, fourcc: str, fps: float, size: tuple[int, int], opened: bool) -> None:
        self.path, self.fourcc, self.fps, self.size = path, fourcc, fps, size
        self.opened = opened
        self.frames: list[np.ndarray] = []
        self.release_calls = 0
        if opened:
            Path(path).write_bytes(b"")  # OpenCV creates the file when it opens it

    def isOpened(self) -> bool:
        return self.opened

    def write(self, image: np.ndarray) -> None:
        assert self.release_calls == 0, "write() after release()"
        self.frames.append(image)

    def release(self) -> None:
        self.release_calls += 1


@pytest.fixture
def fake_cv2(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    state = SimpleNamespace(opened=True, writers=[])

    def video_writer(path: str, fourcc: str, fps: float, size: tuple[int, int]) -> FakeCvWriter:
        writer = FakeCvWriter(path, fourcc, fps, size, state.opened)
        state.writers.append(writer)
        return writer

    module = SimpleNamespace(VideoWriter=video_writer, VideoWriter_fourcc=lambda *chars: "".join(chars))
    monkeypatch.setitem(sys.modules, "cv2", module)
    return state


# --- argument validation ------------------------------------------------------------


@pytest.mark.parametrize("fps", [0, -1, math.nan, math.inf, True, "30", None])
def test_invalid_frame_rate_is_rejected(fake_cv2: SimpleNamespace, tmp_path: Path, fps: Any) -> None:
    with pytest.raises(ValueError, match="fps must be a finite number greater than 0"):
        VideoWriter(tmp_path / "out.mp4", fps=fps, frame_size=SIZE)
    assert fake_cv2.writers == []


@pytest.mark.parametrize("size", [(0, 48), (64, -1), (True, 48), (64.0, 48), (64,), "64x48", None])
def test_invalid_frame_size_is_rejected(fake_cv2: SimpleNamespace, tmp_path: Path, size: Any) -> None:
    with pytest.raises(ValueError, match="frame_size must be"):
        VideoWriter(tmp_path / "out.mp4", fps=30, frame_size=size)


@pytest.mark.parametrize("codec", ["mp4", "mp4vv", "", 42])
def test_invalid_codec_is_rejected(fake_cv2: SimpleNamespace, tmp_path: Path, codec: Any) -> None:
    with pytest.raises(ValueError, match="four-character code"):
        VideoWriter(tmp_path / "out.mp4", fps=30, frame_size=SIZE, codec=codec)


def test_path_without_extension_is_rejected(fake_cv2: SimpleNamespace, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="extension"):
        VideoWriter(tmp_path / "out", fps=30, frame_size=SIZE)


def test_directory_path_is_rejected(fake_cv2: SimpleNamespace, tmp_path: Path) -> None:
    folder = tmp_path / "folder.mp4"
    folder.mkdir()
    with pytest.raises(ValueError, match="directory"):
        VideoWriter(folder, fps=30, frame_size=SIZE)


# --- opening ----------------------------------------------------------------------------


def test_writer_is_opened_with_the_requested_settings(fake_cv2: SimpleNamespace, tmp_path: Path) -> None:
    with VideoWriter(tmp_path / "out.mp4", fps=29.97, frame_size=SIZE) as writer:
        cv_writer = fake_cv2.writers[0]
        assert cv_writer.path == str(tmp_path / "out.partial.mp4")  # extension kept for OpenCV
        assert (cv_writer.fourcc, cv_writer.fps, cv_writer.size) == ("mp4v", 29.97, (64, 48))
        writer.finish()


def test_open_failure_is_reported_and_cleaned_up(fake_cv2: SimpleNamespace, tmp_path: Path) -> None:
    fake_cv2.opened = False
    with pytest.raises(VideoWriteError, match="could not open .* with codec 'mp4v'"):
        VideoWriter(tmp_path / "out.mp4", fps=30, frame_size=SIZE)
    assert fake_cv2.writers[0].release_calls == 1
    assert not (tmp_path / "out.mp4").exists()
    assert not (tmp_path / "out.partial.mp4").exists()


def test_parent_folders_are_created(fake_cv2: SimpleNamespace, tmp_path: Path) -> None:
    with VideoWriter(tmp_path / "a" / "b" / "out.mp4", fps=30, frame_size=SIZE) as writer:
        writer.finish()
    assert (tmp_path / "a" / "b" / "out.mp4").is_file()


def test_existing_output_is_protected(fake_cv2: SimpleNamespace, tmp_path: Path) -> None:
    out = tmp_path / "out.mp4"
    out.write_bytes(b"previous video")
    with pytest.raises(FileExistsError, match="already exists"):
        VideoWriter(out, fps=30, frame_size=SIZE)
    assert out.read_bytes() == b"previous video"


def test_overwrite_replaces_only_on_success(fake_cv2: SimpleNamespace, tmp_path: Path) -> None:
    out = tmp_path / "out.mp4"
    out.write_bytes(b"previous video")
    with VideoWriter(out, fps=30, frame_size=SIZE, overwrite=True) as writer:
        writer.write(frame())  # then abandoned without finish()
    assert out.read_bytes() == b"previous video"

    with VideoWriter(out, fps=30, frame_size=SIZE, overwrite=True) as writer:
        writer.write(frame())
        writer.finish()
    assert out.read_bytes() == b""  # replaced by the fake's (empty) new file


def test_missing_opencv_gives_install_instructions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(MissingDependencyError, match=r'pip install -e "\.\[vision\]"'):
        VideoWriter(tmp_path / "out.mp4", fps=30, frame_size=SIZE)


@pytest.mark.parametrize("size", [(65, 48), (64, 49), (65, 49)])
def test_odd_dimensions_are_rejected_instead_of_silently_cropped(
    fake_cv2: SimpleNamespace, tmp_path: Path, size: tuple[int, int]
) -> None:
    with pytest.raises(VideoWriteError, match="even width and height"):
        VideoWriter(tmp_path / "out.mp4", fps=30, frame_size=size)
    assert fake_cv2.writers == []


# --- writing frames -----------------------------------------------------------------------


def test_frames_are_written_in_order(fake_cv2: SimpleNamespace, tmp_path: Path) -> None:
    frames = [frame(value=v) for v in (10, 20, 30)]
    with VideoWriter(tmp_path / "out.mp4", fps=30, frame_size=SIZE) as writer:
        for f in frames:
            writer.write(f)
        assert writer.frames_written == 3
        writer.finish()
    assert [f[0, 0, 0] for f in fake_cv2.writers[0].frames] == [10, 20, 30]


def test_frame_with_wrong_dimensions_is_rejected(fake_cv2: SimpleNamespace, tmp_path: Path) -> None:
    with VideoWriter(tmp_path / "out.mp4", fps=30, frame_size=SIZE) as writer:
        writer.write(frame())
        with pytest.raises(VideoWriteError, match="Frame 1 is 65 x 48.*64 x 48"):
            writer.write(frame(width=65))
        assert writer.frames_written == 1


@pytest.mark.parametrize(
    "bad",
    [None, np.zeros((48, 64), dtype=np.uint8), np.zeros((48, 64, 3), dtype=np.float32)],
)
def test_invalid_frame_is_rejected(fake_cv2: SimpleNamespace, tmp_path: Path, bad: Any) -> None:
    with VideoWriter(tmp_path / "out.mp4", fps=30, frame_size=SIZE) as writer:
        with pytest.raises(ValueError):
            writer.write(bad)


# --- finishing and cleanup -------------------------------------------------------------------


def test_finish_releases_and_renames(fake_cv2: SimpleNamespace, tmp_path: Path) -> None:
    out = tmp_path / "out.mp4"
    with VideoWriter(out, fps=30, frame_size=SIZE) as writer:
        writer.write(frame())
        assert not out.exists() and (tmp_path / "out.partial.mp4").exists()
        assert writer.finish() == out
    assert fake_cv2.writers[0].release_calls == 1
    assert out.exists() and not (tmp_path / "out.partial.mp4").exists()
    assert writer.closed


def test_error_releases_writer_and_leaves_no_files(fake_cv2: SimpleNamespace, tmp_path: Path) -> None:
    out = tmp_path / "out.mp4"
    with pytest.raises(RuntimeError, match="inference failed"):
        with VideoWriter(out, fps=30, frame_size=SIZE) as writer:
            writer.write(frame())
            raise RuntimeError("inference failed")
    assert fake_cv2.writers[0].release_calls == 1
    assert not out.exists()
    assert not (tmp_path / "out.partial.mp4").exists()


def test_writer_cannot_be_used_after_finish(fake_cv2: SimpleNamespace, tmp_path: Path) -> None:
    with VideoWriter(tmp_path / "out.mp4", fps=30, frame_size=SIZE) as writer:
        writer.finish()
        with pytest.raises(VideoWriteError, match="already closed"):
            writer.write(frame())


def test_partial_path_keeps_the_extension() -> None:
    assert partial_path_for(Path("results/run.mp4")) == Path("results/run.partial.mp4")


# --- frame-rate selection ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "override", "expected"),
    [
        (29.97, None, (29.97, "source")),
        (29.97, 15, (15.0, "override")),  # an explicit value always wins
        (None, None, (FALLBACK_FPS, "fallback")),  # unknown: documented fallback
        (None, 24, (24.0, "override")),
    ],
)
def test_choose_output_fps(source: float | None, override: float | None, expected: tuple) -> None:
    assert choose_output_fps(source, override) == expected


def test_invalid_fps_override_is_rejected() -> None:
    with pytest.raises(ValueError):
        choose_output_fps(30.0, 0)


# --- integration: real OpenCV, real files ----------------------------------------------------------


def test_integration_written_video_can_be_reopened(tmp_path: Path) -> None:
    pytest.importorskip("cv2", reason='OpenCV not installed (pip install -e ".[vision]")')
    from edgevision.video.reader import VideoReader

    out = tmp_path / "real.mp4"
    width, height = SIZE
    try:
        writer = VideoWriter(out, fps=12.5, frame_size=SIZE)
    except VideoWriteError as exc:
        pytest.skip(f"this OpenCV build cannot write mp4v video: {exc}")
    with writer:
        for i in range(7):
            writer.write(frame(width, height, value=i * 30))
        writer.finish()

    with VideoReader(out) as reader:
        frames = list(reader)
        metadata = reader.metadata
    assert len(frames) == 7
    assert (metadata.width, metadata.height) == SIZE
    assert metadata.fps == pytest.approx(12.5)
    assert all(image.shape == (height, width, 3) for _, image in frames)
    # mp4v is lossy, but a flat grey frame should come back close to its original value.
    assert abs(int(frames[3].image.mean()) - 90) <= 3
