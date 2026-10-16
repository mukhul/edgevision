"""Tests for scripts/inspect_video.py, run as a real command in a subprocess."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "inspect_video.py"


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False
    )


def test_help_shows_usage() -> None:
    result = run("--help")
    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "--max-frames" in result.stdout


def test_missing_argument_prints_usage_and_fails() -> None:
    result = run()
    assert result.returncode == 2
    assert "usage:" in result.stderr


def test_missing_video_fails_with_clear_message(tmp_path: Path) -> None:
    result = run(str(tmp_path / "missing.mp4"))
    assert result.returncode == 1
    assert "ERROR: Video file not found" in result.stderr


def test_directory_fails_with_clear_message(tmp_path: Path) -> None:
    result = run(str(tmp_path))
    assert result.returncode == 1
    assert "not a file" in result.stderr


@pytest.mark.parametrize("value", ["0", "-3", "ten"])
def test_invalid_max_frames_is_rejected(tmp_path: Path, value: str) -> None:
    result = run(str(tmp_path / "any.mp4"), "--max-frames", value)
    assert result.returncode == 2
    assert "--max-frames" in result.stderr


def _write_synthetic_video(path: Path, frames: int) -> None:
    cv2 = pytest.importorskip("cv2", reason='OpenCV not installed (pip install -e ".[vision]")')
    np = pytest.importorskip("numpy")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (32, 24))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot encode MJPG video")
    try:
        for _ in range(frames):
            writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    finally:
        writer.release()


def test_reports_metadata_and_end_of_video(tmp_path: Path) -> None:
    video = tmp_path / "clip.avi"
    _write_synthetic_video(video, frames=8)

    result = run(str(video))

    assert result.returncode == 0, result.stderr
    out = " ".join(result.stdout.split())  # ignore column alignment
    assert "Resolution 32 x 24 px" in out
    assert "FPS 25.000" in out
    assert "Frame count 8" in out
    assert "Frames read 8" in out
    assert "end of video (EOF)" in out


def test_reports_frame_limit(tmp_path: Path) -> None:
    video = tmp_path / "clip.avi"
    _write_synthetic_video(video, frames=8)

    result = run(str(video), "--max-frames", "3")

    assert result.returncode == 0, result.stderr
    out = " ".join(result.stdout.split())
    assert "Frames read 3" in out
    assert "--max-frames limit (3) was reached" in out
