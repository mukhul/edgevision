"""Tests for the backend-independent types in edgevision.models.base."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from edgevision.config import load_config
from edgevision.models.base import (
    Detection,
    DetectionSettings,
    validate_bgr_frame,
    validate_frame_index,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
VALID = dict(frame_index=3, bbox_xyxy=(10, 20, 110.5, 220), confidence=0.9, class_id=15, class_name="cat")


def make(**changes: Any) -> Detection:
    return Detection(**{**VALID, **changes})


# --- Detection: valid records --------------------------------------------------


def test_valid_detection() -> None:
    d = make()
    assert d.frame_index == 3
    assert d.bbox_xyxy == (10.0, 20.0, 110.5, 220.0)
    assert all(isinstance(v, float) for v in d.bbox_xyxy)
    assert (d.width, d.height) == (100.5, 200.0)
    assert d.class_name == "cat"


def test_detection_accepts_numpy_numbers_and_converts_them() -> None:
    d = make(
        frame_index=np.int64(4),
        bbox_xyxy=np.array([1, 2, 3, 4], dtype=np.float32),
        confidence=np.float32(0.5),
        class_id=np.int32(0),
    )
    assert type(d.frame_index) is int and type(d.class_id) is int
    assert type(d.confidence) is float
    assert d.bbox_xyxy == (1.0, 2.0, 3.0, 4.0)


def test_zero_size_box_and_boundary_confidences_are_allowed() -> None:
    assert make(bbox_xyxy=(5, 5, 5, 5)).width == 0
    assert make(confidence=0).confidence == 0.0
    assert make(confidence=1).confidence == 1.0


def test_detection_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        make().confidence = 0.1  # type: ignore[misc]


def test_to_dict_is_json_friendly() -> None:
    assert make().to_dict() == {
        "frame_index": 3,
        "bbox_xyxy": [10.0, 20.0, 110.5, 220.0],
        "confidence": 0.9,
        "class_id": 15,
        "class_name": "cat",
    }


# --- Detection: invalid records ------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("frame_index", -1, "frame_index must be a non-negative integer"),
        ("frame_index", True, "frame_index must be a non-negative integer"),
        ("frame_index", 1.0, "frame_index must be a non-negative integer"),
        ("frame_index", "0", "frame_index must be a non-negative integer"),
        ("class_id", -3, "class_id must be a non-negative integer"),
        ("class_id", False, "class_id must be a non-negative integer"),
        ("class_id", 2.5, "class_id must be a non-negative integer"),
        ("confidence", 1.01, "confidence must be a number between 0 and 1"),
        ("confidence", -0.1, "confidence must be a number between 0 and 1"),
        ("confidence", math.nan, "confidence must be a number between 0 and 1"),
        ("confidence", True, "confidence must be a number between 0 and 1"),
        ("confidence", "0.9", "confidence must be a number between 0 and 1"),
        ("class_name", "", "class_name must be a non-empty string"),
        ("class_name", "   ", "class_name must be a non-empty string"),
        ("class_name", 15, "class_name must be a non-empty string"),
        ("class_name", None, "class_name must be a non-empty string"),
        ("bbox_xyxy", (1, 2, 3), "bbox_xyxy must be four numbers"),
        ("bbox_xyxy", (1, 2, 3, 4, 5), "bbox_xyxy must be four numbers"),
        ("bbox_xyxy", "1234", "bbox_xyxy must be four numbers"),
        ("bbox_xyxy", None, "bbox_xyxy must be four numbers"),
        ("bbox_xyxy", (0, 0, math.inf, 10), "finite numbers"),
        ("bbox_xyxy", (0, math.nan, 10, 10), "finite numbers"),
        ("bbox_xyxy", (0, 0, True, 10), "finite numbers"),
        ("bbox_xyxy", (0, 0, "10", 10), "finite numbers"),
        ("bbox_xyxy", (50, 0, 10, 10), "x1 <= x2 and y1 <= y2"),  # x corners swapped
        ("bbox_xyxy", (0, 50, 10, 10), "x1 <= x2 and y1 <= y2"),  # y corners swapped
    ],
)
def test_invalid_detection_is_rejected(field: str, value: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make(**{field: value})


# --- DetectionSettings -----------------------------------------------------------


def test_default_settings_match_default_config() -> None:
    model = load_config(DEFAULT_CONFIG).model
    assert DetectionSettings() == DetectionSettings.from_model_config(model)


def test_settings_from_config_values() -> None:
    s = DetectionSettings(input_size=320, conf_threshold=1, iou_threshold=0.5, classes=[0, 15])
    assert s.conf_threshold == 1.0 and isinstance(s.conf_threshold, float)
    assert s.classes == (0, 15)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"input_size": 0}, "input_size must be an integer >= 32"),
        ({"input_size": 641}, "multiple of 32"),
        ({"input_size": 640.0}, "input_size must be an integer"),
        ({"input_size": "640"}, "input_size must be an integer"),
        ({"input_size": True}, "input_size must be an integer"),
        ({"conf_threshold": 1.5}, "conf_threshold must be a number between 0 and 1"),
        ({"conf_threshold": -0.01}, "conf_threshold must be a number between 0 and 1"),
        ({"conf_threshold": True}, "conf_threshold must be a number between 0 and 1"),
        ({"conf_threshold": math.nan}, "conf_threshold must be a number between 0 and 1"),
        ({"iou_threshold": 2}, "iou_threshold must be a number between 0 and 1"),
        ({"iou_threshold": "0.7"}, "iou_threshold must be a number between 0 and 1"),
        ({"classes": []}, "classes must be None"),
        ({"classes": [-1]}, "classes must be None"),
        ({"classes": [True]}, "classes must be None"),
        ({"classes": "0"}, "classes must be None"),
        ({"classes": 0}, "classes must be None"),
    ],
)
def test_invalid_settings_are_rejected(changes: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DetectionSettings(**changes)


# --- frame validation ------------------------------------------------------------


def test_valid_bgr_frame_passes() -> None:
    validate_bgr_frame(np.zeros((48, 64, 3), dtype=np.uint8))


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (None, "must be a NumPy array"),
        ([[[0, 0, 0]]], "must be a NumPy array"),
        (np.zeros((0, 0, 3), dtype=np.uint8), "frame is empty"),
        (np.zeros((48, 0, 3), dtype=np.uint8), "frame is empty"),
        (np.zeros((48, 64), dtype=np.uint8), r"shape \(height, width, 3\)"),  # grayscale
        (np.zeros((48, 64, 4), dtype=np.uint8), r"shape \(height, width, 3\)"),  # BGRA
        (np.zeros(10, dtype=np.uint8), r"shape \(height, width, 3\)"),
        (np.zeros((48, 64, 3), dtype=np.float32), "dtype uint8"),
    ],
)
def test_invalid_frames_are_rejected(frame: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_bgr_frame(frame)


@pytest.mark.parametrize("value", [-1, True, 2.0, None])
def test_invalid_frame_index_is_rejected(value: Any) -> None:
    with pytest.raises(ValueError, match="frame_index"):
        validate_frame_index(value)
