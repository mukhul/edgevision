"""Tests for edgevision.video.annotator.

Pure helpers (colours, clipping, label text) are tested without OpenCV.
Drawing tests use real OpenCV on small synthetic frames and inspect pixels,
plus a spy on cv2.putText to check exactly which labels were drawn.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from edgevision.models.base import Detection
from edgevision.tracking.base import TrackedDetection
from edgevision.video.annotator import (
    PALETTE,
    annotate_frame,
    class_color,
    clip_box,
    label_text,
)

H, W = 120, 160


def blank() -> np.ndarray:
    return np.full((H, W, 3), 30, dtype=np.uint8)  # dark grey, unlike any palette colour


def det(box: tuple[float, float, float, float], class_id: int = 15, name: str = "cat", conf: float = 0.91) -> Detection:
    return Detection(0, box, conf, class_id, name)


@pytest.fixture
def cv2() -> Any:
    return pytest.importorskip("cv2", reason='OpenCV not installed (pip install -e ".[vision]")')


@pytest.fixture
def drawn_labels(cv2: Any, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the text of every cv2.putText call while still drawing it."""
    labels: list[str] = []
    real_put_text = cv2.putText

    def spy(img: Any, text: str, *args: Any, **kwargs: Any) -> Any:
        labels.append(text)
        return real_put_text(img, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", spy)
    return labels


# --- pure helpers (no OpenCV needed) ------------------------------------------------


def test_class_colours_are_deterministic_and_distinct() -> None:
    assert class_color(15) == class_color(15)
    assert class_color(0) != class_color(15)
    assert len(set(PALETTE)) == len(PALETTE)
    assert class_color(len(PALETTE) + 3) == class_color(3)  # palette repeats predictably
    assert all(0 <= c <= 255 for colour in PALETTE for c in colour)


def test_label_shows_class_name_and_confidence() -> None:
    assert label_text(det((0, 0, 1, 1), name="potted plant", conf=0.8)) == "potted plant 0.80"


def test_label_shows_track_id_for_tracked_detections() -> None:
    tracked = TrackedDetection.from_detection(det((0, 0, 1, 1), class_id=0, name="person", conf=0.92), 3)
    assert label_text(tracked) == "person #3 0.92"


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        ((10.4, 20.6, 50.5, 60.2), (10, 21, 50, 60)),  # rounded to whole pixels
        ((-30, -5, 500, 900), (0, 0, W - 1, H - 1)),  # clipped to the frame
        ((W - 1, H - 1, W + 50, H + 50), (W - 1, H - 1, W - 1, H - 1)),  # touches the corner
        ((W + 10, 0, W + 50, 10), None),  # completely right of the frame
        ((0, -50, 10, -1), None),  # completely above the frame
    ],
)
def test_clip_box(box: tuple, expected: tuple | None) -> None:
    assert clip_box(box, W, H) == expected


# --- drawing --------------------------------------------------------------------------


def test_frame_without_detections_is_an_unchanged_copy(cv2: Any) -> None:
    frame = blank()
    out = annotate_frame(frame, [])
    assert out is not frame
    assert np.array_equal(out, frame)


def test_one_detection_draws_box_and_label(cv2: Any, drawn_labels: list[str]) -> None:
    frame = blank()
    original = frame.copy()
    out = annotate_frame(frame, [det((40, 50, 120, 110))])

    assert np.array_equal(frame, original)  # the input frame was not modified
    assert drawn_labels == ["cat 0.91"]
    colour = np.array(class_color(15))
    # Box edges are drawn in the class colour (sample the middle of the left and bottom edges).
    assert np.array_equal(out[80, 40], colour)
    assert np.array_equal(out[110, 80], colour)
    # The middle of the box is untouched: only the outline is drawn.
    assert np.array_equal(out[80, 80], original[80, 80])
    # The filled label sits above the box.
    assert (out[35:50, 41:60] != original[35:50, 41:60]).any()


def test_multiple_detections_get_their_own_labels_and_colours(cv2: Any, drawn_labels: list[str]) -> None:
    detections = [
        det((10, 40, 60, 100), class_id=15, name="cat", conf=0.9),
        det((90, 40, 150, 100), class_id=0, name="person", conf=0.55),
    ]
    out = annotate_frame(blank(), detections)
    assert drawn_labels == ["cat 0.90", "person 0.55"]
    assert np.array_equal(out[70, 10], np.array(class_color(15)))
    assert np.array_equal(out[70, 90], np.array(class_color(0)))


def test_tracked_detection_is_drawn_like_a_detection_plus_id(cv2: Any, drawn_labels: list[str]) -> None:
    plain = det((40, 50, 120, 110))
    tracked = TrackedDetection.from_detection(plain, 7)
    a = annotate_frame(blank(), [plain])
    b = annotate_frame(blank(), [tracked])
    assert drawn_labels == ["cat 0.91", "cat #7 0.91"]
    assert np.array_equal(a[50:111, 40:121], b[50:111, 40:121])  # same box, same class colour


def test_same_class_gets_the_same_colour_on_every_frame(cv2: Any) -> None:
    a = annotate_frame(blank(), [det((10, 40, 60, 100))])
    b = annotate_frame(blank(), [det((10, 40, 60, 100))])
    assert np.array_equal(a, b)  # drawing is fully deterministic


def test_boxes_beyond_the_frame_are_clipped_not_rejected(cv2: Any) -> None:
    out = annotate_frame(blank(), [det((-40, -40, W + 40, H + 40))])  # larger than the frame
    colour = np.array(class_color(15))
    assert np.array_equal(out[H // 2, 0], colour)  # left edge drawn on the frame border
    assert np.array_equal(out[H - 1, W // 2], colour)  # bottom edge drawn on the frame border


def test_box_entirely_outside_the_frame_draws_nothing(cv2: Any, drawn_labels: list[str]) -> None:
    frame = blank()
    out = annotate_frame(frame, [det((W + 5, 10, W + 40, 50))])
    assert np.array_equal(out, frame)
    assert drawn_labels == []


def test_label_of_a_box_at_the_top_edge_stays_inside_the_frame(cv2: Any) -> None:
    out = annotate_frame(blank(), [det((100, 0, W - 1, 60), name="potted plant")])
    assert (out[0:10, 100:W] != 30).any()  # label drawn inside, at the top


def test_tiny_frame_is_handled(cv2: Any) -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    assert annotate_frame(frame, [det((1, 1, 6, 6))]).shape == (8, 8, 3)


@pytest.mark.parametrize(
    "frame",
    [None, np.zeros((H, W), dtype=np.uint8), np.zeros((H, W, 3), dtype=np.float32), np.zeros((0, W, 3), dtype=np.uint8)],
)
def test_invalid_frames_are_rejected(frame: Any) -> None:
    with pytest.raises(ValueError):
        annotate_frame(frame, [])


def test_non_detection_items_are_rejected() -> None:
    with pytest.raises(TypeError, match="Detection"):
        annotate_frame(blank(), [{"bbox_xyxy": (0, 0, 1, 1)}])  # type: ignore[list-item]
