"""Tests for edgevision.results (streaming, atomic JSON output)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edgevision.models.base import Detection
from edgevision.pipeline import FrameDetections
from edgevision.results import SCHEMA_NAME, SCHEMA_VERSION, VideoDetectionsWriter, video_header
from edgevision.video.reader import VideoMetadata

SUMMARY = {"frames_processed": 2, "total_detections": 1}


def header() -> dict:
    metadata = VideoMetadata(Path("data/input/clip.mp4"), 640, 480, None, 2, "avc1")
    return video_header(
        metadata=metadata,
        backend="pytorch",
        model="models/yolov8n.pt",
        device="cpu",
        settings={"input_size": 640, "conf_threshold": 0.25, "iou_threshold": 0.7, "classes": None},
    )


FRAMES = [
    FrameDetections(0, (Detection(0, (1, 2, 30, 40), 0.9, 15, "cat"),)),
    FrameDetections(1, ()),
]


def write(path: Path, **kwargs) -> Path:
    with VideoDetectionsWriter(path, header(), **kwargs) as writer:
        for frame in FRAMES:
            writer.write_frame(frame)
        return writer.finish(SUMMARY)


def test_output_follows_the_documented_schema(tmp_path: Path) -> None:
    doc = json.loads(write(tmp_path / "out.json").read_text(encoding="utf-8"))  # standard json parser

    assert list(doc) == [
        "schema", "schema_version", "edgevision_version", "created_utc",
        "video", "detector", "bbox_format", "frames", "summary",
    ]
    assert (doc["schema"], doc["schema_version"]) == (SCHEMA_NAME, SCHEMA_VERSION)
    assert doc["video"] == {
        "path": "data/input/clip.mp4", "width": 640, "height": 480,
        "fps": None, "frame_count_reported": 2, "fourcc": "avc1",
    }
    assert doc["detector"]["backend"] == "pytorch" and doc["detector"]["settings"]["input_size"] == 640
    assert doc["bbox_format"] == "xyxy_pixels"
    assert doc["frames"] == [
        {"frame_index": 0, "detections": [
            {"bbox_xyxy": [1.0, 2.0, 30.0, 40.0], "confidence": 0.9, "class_id": 15, "class_name": "cat"}
        ]},
        {"frame_index": 1, "detections": []},
    ]
    assert doc["summary"] == SUMMARY
    assert doc["created_utc"].endswith("Z")


def test_detection_only_header_has_no_tracker_key() -> None:
    assert "tracker" not in header()


def test_tracked_output_includes_tracker_and_track_ids(tmp_path: Path) -> None:
    from edgevision.tracking.base import TrackedDetection

    tracked_header = {**header(), "tracker": {"name": "bytetrack", "config": "x.yaml", "settings": {"track_buffer": 30}}}
    frames = [FrameDetections(0, (
        TrackedDetection(0, (1, 2, 30, 40), 0.9, 15, "cat", 3),
        Detection(0, (50, 50, 60, 60), 0.3, 0, "person"),  # not assigned to a track
    ), tracked=True)]
    path = tmp_path / "tracked.json"
    with VideoDetectionsWriter(path, tracked_header) as writer:
        writer.write_frame(frames[0])
        writer.finish(SUMMARY)

    doc = json.loads(path.read_text(encoding="utf-8"))
    assert list(doc)[7] == "tracker"
    assert doc["tracker"]["name"] == "bytetrack"
    assert [d["track_id"] for d in doc["frames"][0]["detections"]] == [3, None]
    assert all(type(d["track_id"]) in (int, type(None)) for d in doc["frames"][0]["detections"])


def test_video_with_no_frames_is_still_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    with VideoDetectionsWriter(path, header()) as writer:
        writer.finish({"frames_processed": 0})
    assert json.loads(path.read_text(encoding="utf-8"))["frames"] == []


def test_file_only_appears_when_finished(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    with VideoDetectionsWriter(path, header()) as writer:
        writer.write_frame(FRAMES[0])
        assert not path.exists()  # still being written...
        assert writer.partial_path.exists()  # ...to the .partial file
        writer.finish(SUMMARY)
    assert path.exists()
    assert not writer.partial_path.exists()


def test_error_leaves_no_results_file(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    with pytest.raises(RuntimeError, match="crash"):
        with VideoDetectionsWriter(path, header()) as writer:
            writer.write_frame(FRAMES[0])
            raise RuntimeError("crash")
    assert not path.exists()
    assert not writer.partial_path.exists()


def test_leaving_without_finish_leaves_no_results_file(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    with VideoDetectionsWriter(path, header()) as writer:
        writer.write_frame(FRAMES[0])
    assert not path.exists() and not writer.partial_path.exists()


def test_existing_file_is_protected(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    path.write_text("old results", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        VideoDetectionsWriter(path, header())
    assert path.read_text(encoding="utf-8") == "old results"


def test_overwrite_replaces_only_on_success(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    path.write_text("old results", encoding="utf-8")

    with VideoDetectionsWriter(path, header(), overwrite=True) as writer:
        writer.write_frame(FRAMES[0])  # then abandoned without finish()
    assert path.read_text(encoding="utf-8") == "old results"  # untouched by the failed run

    write(path, overwrite=True)
    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == SCHEMA_NAME


def test_missing_parent_folders_are_created(tmp_path: Path) -> None:
    path = write(tmp_path / "a" / "b" / "out.json")
    assert path.is_file()


def test_directory_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(IsADirectoryError):
        VideoDetectionsWriter(tmp_path, header())


def test_non_json_values_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        with VideoDetectionsWriter(tmp_path / "out.json", header()) as writer:
            writer.finish({"value": float("nan")})  # NaN is not valid JSON


def test_writer_cannot_be_used_after_finish(tmp_path: Path) -> None:
    with VideoDetectionsWriter(tmp_path / "out.json", header()) as writer:
        writer.finish(SUMMARY)
        with pytest.raises(ValueError, match="already finished"):
            writer.write_frame(FRAMES[0])
