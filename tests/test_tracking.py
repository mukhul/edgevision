"""Tests for edgevision.tracking.

* TrackedDetection, config validation and ID attachment are tested as pure
  Python, without Ultralytics.
* ByteTrackTracker is tested with the *real* Ultralytics ByteTrack on
  hand-made detections (moving boxes). No model, weights, video or GPU is
  involved, so these are fast and deterministic. They are skipped only if the
  `tracking` extra is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from edgevision.models.base import Detection
from edgevision.optional import MissingDependencyError
from edgevision.tracking.base import TrackedDetection, Tracker, TrackingError
from edgevision.tracking.bytetrack import (
    ByteTrackTracker,
    attach_track_ids,
    validate_bytetrack_settings,
)

FRAME = np.zeros((240, 320, 3), dtype=np.uint8)
GOOD_SETTINGS = {
    "tracker_type": "bytetrack", "track_high_thresh": 0.25, "track_low_thresh": 0.1,
    "new_track_thresh": 0.25, "track_buffer": 30, "match_thresh": 0.8, "fuse_score": True,
}


def det(frame_index: int, box: tuple, conf: float = 0.9, class_id: int = 0, name: str = "person") -> Detection:
    return Detection(frame_index, box, conf, class_id, name)


def ids(detections: list[Detection]) -> list[int | None]:
    return [getattr(d, "track_id", None) for d in detections]


# --- TrackedDetection ------------------------------------------------------------------


def test_tracked_detection_keeps_detection_fields_and_adds_id() -> None:
    base = det(4, (1, 2, 30, 40), 0.8, 15, "cat")
    tracked = TrackedDetection.from_detection(base, 7)
    assert isinstance(tracked, Detection)  # usable anywhere a Detection is
    assert (tracked.frame_index, tracked.bbox_xyxy, tracked.confidence, tracked.class_id, tracked.class_name) == (
        4, (1.0, 2.0, 30.0, 40.0), 0.8, 15, "cat"
    )
    assert tracked.track_id == 7
    assert tracked.to_dict() == {**base.to_dict(), "track_id": 7}
    assert tracked != base  # a tracked and an untracked detection are different things


@pytest.mark.parametrize("track_id", [-1, True, 1.5, "3", None])
def test_invalid_track_id_is_rejected(track_id: Any) -> None:
    with pytest.raises(ValueError, match="track_id must be a non-negative integer"):
        TrackedDetection(0, (0, 0, 1, 1), 0.5, 0, "person", track_id)


def test_tracked_detection_still_validates_the_detection() -> None:
    with pytest.raises(ValueError, match="confidence"):
        TrackedDetection(0, (0, 0, 1, 1), 1.5, 0, "person", 1)


def test_numpy_track_id_becomes_a_plain_int() -> None:
    assert type(TrackedDetection(0, (0, 0, 1, 1), 0.5, 0, "person", np.int64(3)).track_id) is int


# --- attaching IDs (pure logic) --------------------------------------------------------


def test_attach_track_ids_keeps_order_and_unmatched_detections() -> None:
    detections = [det(0, (0, 0, 5, 5), class_id=0), det(0, (10, 10, 20, 20), class_id=2, name="car"), det(0, (30, 30, 40, 40))]
    result = attach_track_ids(detections, [(2, 9, 0), (0, 4, 0)])  # tracker order differs from input order
    assert ids(result) == [4, None, 9]
    assert result[1] is detections[1]  # unmatched detection returned unchanged


@pytest.mark.parametrize(
    ("matches", "message"),
    [
        ([(3, 1, 0)], "referred to detection 3"),
        ([(-1, 1, 0)], "referred to detection -1"),
        ([(0, 1, 0), (0, 2, 0)], "more than once"),
        ([(0, 1, 0), (1, 1, 0)], "more than once"),
        ([(0, 1, 5)], "reported class 5"),
    ],
)
def test_inconsistent_tracker_output_is_rejected(matches: list, message: str) -> None:
    detections = [det(0, (0, 0, 5, 5)), det(0, (10, 10, 20, 20))]
    with pytest.raises(TrackingError, match=message):
        attach_track_ids(detections, matches)


# --- configuration -------------------------------------------------------------------


def test_valid_settings_are_accepted() -> None:
    assert validate_bytetrack_settings(dict(GOOD_SETTINGS), "test") == GOOD_SETTINGS


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"tracker_type": "botsort"}, "only 'bytetrack' is supported"),
        ({"track_high_thresh": 1.5}, "'track_high_thresh' must be a number between 0 and 1"),
        ({"match_thresh": "0.8"}, "'match_thresh' must be a number"),
        ({"track_buffer": 0}, "'track_buffer' must be a positive integer"),
        ({"track_buffer": 2.5}, "'track_buffer' must be a positive integer"),
        ({"fuse_score": 1}, "'fuse_score' must be true or false"),
    ],
)
def test_invalid_settings_are_rejected(change: dict, message: str) -> None:
    with pytest.raises(TrackingError, match=message):
        validate_bytetrack_settings({**GOOD_SETTINGS, **change}, "test")


def test_missing_setting_is_reported() -> None:
    settings = dict(GOOD_SETTINGS)
    del settings["track_buffer"]
    with pytest.raises(TrackingError, match="missing 'track_buffer'"):
        validate_bytetrack_settings(settings, "test")


# --- missing optional dependencies ---------------------------------------------------


@pytest.mark.parametrize(("module", "name"), [("lap", "lap"), ("ultralytics", "Ultralytics")])
def test_missing_dependency_gives_install_instructions(
    monkeypatch: pytest.MonkeyPatch, module: str, name: str
) -> None:
    # Provide a stand-in `lap` so each case tests exactly one missing package,
    # whatever is installed on this machine.
    monkeypatch.setitem(sys.modules, "lap", SimpleNamespace(__version__="0.5.13"))
    monkeypatch.setitem(sys.modules, module, None)  # behaves like "not installed"
    with pytest.raises(MissingDependencyError) as info:
        ByteTrackTracker()
    assert f"{name} is required to run ByteTrack tracking" in str(info.value)
    assert 'pip install -e ".[tracking]"' in str(info.value)


# --- real ByteTrack on hand-made detections --------------------------------------------


@pytest.fixture
def tracker() -> ByteTrackTracker:
    pytest.importorskip("lap", reason='tracking extra not installed (pip install -e ".[tracking]")')
    pytest.importorskip("ultralytics", reason='tracking extra not installed (pip install -e ".[tracking]")')
    return ByteTrackTracker()


def write_config(tmp_path: Path, **changes: Any) -> Path:
    import yaml

    path = tmp_path / "bytetrack_custom.yaml"
    path.write_text(yaml.safe_dump({**GOOD_SETTINGS, **changes}), encoding="utf-8")
    return path


def test_tracker_satisfies_the_protocol(tracker: ByteTrackTracker) -> None:
    assert isinstance(tracker, Tracker)
    assert tracker.name == "bytetrack"
    assert tracker.settings["tracker_type"] == "bytetrack"


def test_moving_objects_keep_their_ids(tracker: ByteTrackTracker) -> None:
    for i in range(6):
        frame_dets = [
            det(i, (10 + 6 * i, 20, 60 + 6 * i, 90)),  # person moving right
            det(i, (200, 100 + 4 * i, 260, 160 + 4 * i), 0.8, 15, "cat"),  # cat moving down
        ]
        result = tracker.update(frame_dets, FRAME)
        assert ids(result) == [1, 2], f"frame {i}"


def test_original_boxes_and_scores_are_kept(tracker: ByteTrackTracker) -> None:
    detections = [det(0, (11.5, 22.25, 61.75, 90.5), 0.87)]
    (tracked,) = tracker.update(detections, FRAME)
    assert tracked.bbox_xyxy == detections[0].bbox_xyxy  # not the tracker's smoothed box
    assert tracked.confidence == detections[0].confidence


def test_frames_without_detections(tracker: ByteTrackTracker) -> None:
    assert tracker.update([], FRAME) == []
    # After the first frame, a new object is only confirmed when seen a second time.
    assert ids(tracker.update([det(1, (10, 10, 50, 50))], FRAME)) == [None]
    assert ids(tracker.update([det(2, (11, 10, 51, 50))], FRAME)) == [1]
    assert tracker.update([], FRAME) == []


def test_briefly_missed_object_keeps_its_id(tracker: ByteTrackTracker) -> None:
    for i in range(3):
        tracker.update([det(i, (100, 100, 150, 160))], FRAME)
    for i in range(3, 6):
        tracker.update([], FRAME)  # missed for 3 frames, within track_buffer
    assert ids(tracker.update([det(6, (102, 100, 152, 160))], FRAME)) == [1]


def test_object_lost_longer_than_buffer_gets_a_new_id(tmp_path: Path) -> None:
    pytest.importorskip("lap", reason='tracking extra not installed (pip install -e ".[tracking]")')
    pytest.importorskip("ultralytics", reason='tracking extra not installed (pip install -e ".[tracking]")')
    tracker = ByteTrackTracker(write_config(tmp_path, track_buffer=2))
    for i in range(3):
        tracker.update([det(i, (100, 100, 150, 160))], FRAME)
    for i in range(3, 8):
        tracker.update([], FRAME)  # missed for 5 frames, longer than track_buffer=2
    tracker.update([det(8, (100, 100, 150, 160))], FRAME)
    second = tracker.update([det(9, (100, 100, 150, 160))], FRAME)
    assert ids(second) == [2]  # same place, but the old track was removed: a new ID


def test_low_score_detection_continues_a_track_but_does_not_start_one(tracker: ByteTrackTracker) -> None:
    tracker.update([det(0, (100, 100, 150, 160), 0.9)], FRAME)
    # Frame 1: the known object is seen only weakly (0.15); a new weak object appears elsewhere.
    result = tracker.update([det(1, (101, 100, 151, 160), 0.15), det(1, (10, 10, 40, 40), 0.15)], FRAME)
    assert ids(result) == [1, None]  # ByteTrack's second pass keeps track 1; no new track


def test_reset_forgets_tracks_and_restarts_ids(tracker: ByteTrackTracker) -> None:
    for i in range(3):
        tracker.update([det(i, (10, 10, 50, 50)), det(i, (100, 100, 150, 150))], FRAME)
    tracker.reset()
    assert ids(tracker.update([det(0, (200, 200, 250, 230))], FRAME)) == [1]


def test_custom_config_file_is_used(tmp_path: Path) -> None:
    pytest.importorskip("lap", reason='tracking extra not installed (pip install -e ".[tracking]")')
    pytest.importorskip("ultralytics", reason='tracking extra not installed (pip install -e ".[tracking]")')
    path = write_config(tmp_path, track_buffer=7)
    tracker = ByteTrackTracker(path)
    assert tracker.settings["track_buffer"] == 7
    assert tracker.config_source == str(path)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("tracker_type: botsort\n", "only 'bytetrack' is supported"),
        ("tracker_type: [unclosed\n", "Could not read tracker config"),
        ("- just\n- a list\n", "must be a mapping"),
    ],
)
def test_bad_config_file_is_rejected(tmp_path: Path, content: str, message: str) -> None:
    pytest.importorskip("lap", reason='tracking extra not installed (pip install -e ".[tracking]")')
    pytest.importorskip("ultralytics", reason='tracking extra not installed (pip install -e ".[tracking]")')
    path = tmp_path / "bad.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(TrackingError, match=message):
        ByteTrackTracker(path)


def test_missing_config_file_is_rejected(tmp_path: Path) -> None:
    pytest.importorskip("lap", reason='tracking extra not installed (pip install -e ".[tracking]")')
    pytest.importorskip("ultralytics", reason='tracking extra not installed (pip install -e ".[tracking]")')
    with pytest.raises(TrackingError, match="not found"):
        ByteTrackTracker(tmp_path / "missing.yaml")


def test_tracker_errors_are_not_swallowed(tracker: ByteTrackTracker, monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*args: Any, **kwargs: Any) -> None:
        raise ValueError("matrix shape mismatch")

    monkeypatch.setattr(tracker._backend, "update", explode)
    with pytest.raises(TrackingError, match="ByteTrack update failed: ValueError: matrix shape mismatch") as info:
        tracker.update([det(0, (0, 0, 10, 10))], FRAME)
    assert isinstance(info.value.__cause__, ValueError)


def test_invalid_inputs_are_rejected(tracker: ByteTrackTracker) -> None:
    with pytest.raises(ValueError):
        tracker.update([], np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(TypeError, match="Detection"):
        tracker.update([{"bbox": (0, 0, 1, 1)}], FRAME)  # type: ignore[list-item]
