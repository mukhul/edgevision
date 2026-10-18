"""Tests for edgevision.evaluation.annotations (COCO JSON ground truth). Synthetic files only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from edgevision.evaluation.annotations import AnnotationError, load_coco  # noqa: E402

MODEL_NAMES = {0: "person", 2: "car", 15: "cat"}


def coco(**changes) -> dict:
    data = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 640, "height": 480},
                   {"id": 7, "file_name": "b.jpg", "width": 320, "height": 240}],
        "categories": [{"id": 1, "name": "person"}, {"id": 3, "name": "Car"}, {"id": 99, "name": "unicorn"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40], "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 3, "bbox": [100.5, 50, 20, 10]},
            {"id": 3, "image_id": 1, "category_id": 1, "bbox": [0, 0, 300, 200], "iscrowd": 1},
            {"id": 4, "image_id": 7, "category_id": 99, "bbox": [1, 1, 5, 5]},
        ],
    }
    data.update(changes)
    return data


def write(tmp_path: Path, data) -> Path:
    path = tmp_path / "annotations.json"
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return path


def test_valid_file(tmp_path: Path) -> None:
    ds = load_coco(write(tmp_path, coco()), MODEL_NAMES)
    assert [i.file_name for i in ds.images] == ["a.jpg", "b.jpg"]
    a = ds.boxes[1]
    assert a[0].bbox_xyxy == (10, 20, 40, 60) and a[0].class_id == 0 and not a[0].crowd  # x, y, w, h -> corners
    assert a[1].bbox_xyxy == (100.5, 50, 120.5, 60) and a[1].class_id == 2  # "Car" matched case-insensitively
    assert a[2].crowd
    assert ds.boxes[7] == ()  # an image whose only label is unknown to the model
    assert ds.class_names == {0: "person", 2: "car"}  # only classes present in the categories
    assert ds.unmapped_categories == ("unicorn",) and ds.dropped_annotations == 1
    assert ds.num_objects == 2  # crowd regions are not objects to find
    assert "unicorn" in ds.warnings[0]


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ("{not json", "not valid JSON"),
        ([], "must contain a JSON object"),
        ({"images": []}, "missing the list"),
        (coco(images=[]), "no images"),
        (coco(categories=[{"id": 1, "name": "unicorn"}],
              annotations=[]), "none of the category names match"),
        (coco(images=[{"id": 1, "file_name": "a.jpg", "width": 0, "height": 480}]), "positive integer 'width'"),
        (coco(images=[{"id": 1, "file_name": "a.jpg", "width": 9, "height": 9}] * 2, annotations=[]),
         "duplicate image id"),
        (coco(categories=[{"id": 1, "name": "person"}, {"id": 1, "name": "car"}], annotations=[]),
         "duplicate category id"),
        (coco(annotations=[{"image_id": 5, "category_id": 1, "bbox": [0, 0, 1, 1]}]), "unknown image_id 5"),
        (coco(annotations=[{"image_id": 1, "category_id": 42, "bbox": [0, 0, 1, 1]}]), "unknown category_id 42"),
        (coco(annotations=[{"image_id": 1, "category_id": 1, "bbox": [0, 0, 1]}]), "four finite numbers"),
        (coco(annotations=[{"image_id": 1, "category_id": 1, "bbox": [0, 0, "1", 1]}]), "four finite numbers"),
        (coco(annotations=[{"image_id": 1, "category_id": 1, "bbox": [0, 0, 0, 5]}]), "width 0.0"),
        (coco(annotations=[{"image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5], "iscrowd": 2}]), "iscrowd 2"),
    ],
)
def test_invalid_files(tmp_path: Path, data, message: str) -> None:
    with pytest.raises(AnnotationError, match=message):
        load_coco(write(tmp_path, data), MODEL_NAMES)


def test_all_problems_are_reported_together(tmp_path: Path) -> None:
    data = coco(annotations=[{"image_id": 5, "category_id": 1, "bbox": [0, 0, 1, 1]},
                             {"image_id": 1, "category_id": 1, "bbox": [0, 0, -1, 1]}])
    with pytest.raises(AnnotationError, match=r"2 problem\(s\)"):
        load_coco(write(tmp_path, data), MODEL_NAMES)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AnnotationError, match="not found"):
        load_coco(tmp_path / "missing.json", MODEL_NAMES)


def test_boxes_outside_the_image_are_kept_with_a_warning(tmp_path: Path) -> None:
    data = coco(annotations=[{"image_id": 7, "category_id": 1, "bbox": [300, 200, 50, 50]}], categories=[
        {"id": 1, "name": "person"}])
    ds = load_coco(write(tmp_path, data), MODEL_NAMES)
    assert len(ds.boxes[7]) == 1 and any("outside image b.jpg" in w for w in ds.warnings)
