"""Load labelled evaluation data (ground truth) in COCO JSON format.

The format is the standard COCO "instances" layout, which most labelling
tools can export (CVAT, Label Studio, Roboflow, makesense.ai...):

.. code-block:: json

    {
      "images":      [{"id": 1, "file_name": "street_001.jpg", "width": 1280, "height": 720}],
      "categories":  [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}],
      "annotations": [{"id": 10, "image_id": 1, "category_id": 3, "bbox": [x, y, width, height], "iscrowd": 0}]
    }

* ``bbox`` is ``[x, y, width, height]`` in pixels, ``(x, y)`` = top-left corner
  (COCO convention). It is converted to EdgeVision's ``x1, y1, x2, y2``.
* Categories are matched to the model's classes **by name**
  (case-insensitive), because COCO category IDs (1-90 with gaps) differ from
  the model's class IDs (0-79). Categories whose name the model does not know
  are reported and their annotations are not evaluated.
* ``iscrowd: 1`` marks a region containing many objects; predictions inside
  it count neither as correct nor as wrong. Segmentation fields are ignored.
* Image files are found by ``file_name`` relative to an images folder.

Nothing here invents labels: every evaluated object comes from the file.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from edgevision.evaluation.detection_metrics import GroundTruthBox


class AnnotationError(ValueError):
    """The annotation file is missing, malformed or does not fit the model."""


@dataclass(frozen=True)
class LabelledImage:
    image_id: int
    file_name: str
    width: int
    height: int


@dataclass(frozen=True)
class GroundTruthDataset:
    """Labelled images with their boxes, in the model's class IDs."""

    images: tuple[LabelledImage, ...]
    boxes: dict[int, tuple[GroundTruthBox, ...]]  # image_id -> boxes (possibly empty)
    class_names: dict[int, str]  # model class ID -> name, for the classes present in the categories
    unmapped_categories: tuple[str, ...] = ()
    dropped_annotations: int = 0  # annotations of unmapped categories
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def num_objects(self) -> int:
        return sum(1 for boxes in self.boxes.values() for b in boxes if not b.crowd)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_coco(path: str | os.PathLike[str], model_class_names: dict[int, str]) -> GroundTruthDataset:
    """Read a COCO JSON file and map its categories to ``model_class_names`` by name.

    Raises AnnotationError with every problem found (not just the first).
    """
    path = Path(path)
    if not path.is_file():
        raise AnnotationError(f"Annotation file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AnnotationError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise AnnotationError(f"{path} must contain a JSON object with 'images', 'annotations' and 'categories'")
    missing = [k for k in ("images", "annotations", "categories") if not isinstance(data.get(k), list)]
    if missing:
        raise AnnotationError(f"{path} is missing the list(s) {missing} (expected COCO instances format)")

    problems: list[str] = []
    by_name = {name.strip().lower(): int(cid) for cid, name in model_class_names.items()}

    category_to_class: dict[int, int] = {}
    unmapped: list[str] = []
    for i, cat in enumerate(data["categories"]):
        if not isinstance(cat, dict) or not _is_int(cat.get("id")) or not isinstance(cat.get("name"), str):
            problems.append(f"categories[{i}] needs an integer 'id' and a string 'name'")
            continue
        if cat["id"] in category_to_class:
            problems.append(f"categories[{i}]: duplicate category id {cat['id']}")
            continue
        class_id = by_name.get(cat["name"].strip().lower())
        if class_id is None:
            unmapped.append(cat["name"])
            category_to_class[cat["id"]] = -1
        else:
            category_to_class[cat["id"]] = class_id
    if category_to_class and all(v == -1 for v in category_to_class.values()):
        problems.append("none of the category names match the model's class names (e.g. 'person', 'car')")

    images: dict[int, LabelledImage] = {}
    for i, img in enumerate(data["images"]):
        if not isinstance(img, dict) or not _is_int(img.get("id")) or not isinstance(img.get("file_name"), str):
            problems.append(f"images[{i}] needs an integer 'id' and a string 'file_name'")
            continue
        if not (_is_int(img.get("width")) and _is_int(img.get("height")) and img["width"] > 0 and img["height"] > 0):
            problems.append(f"images[{i}] ({img['file_name']}) needs positive integer 'width' and 'height'")
            continue
        if img["id"] in images:
            problems.append(f"images[{i}]: duplicate image id {img['id']}")
            continue
        images[img["id"]] = LabelledImage(img["id"], img["file_name"], img["width"], img["height"])
    if not images and not problems:
        problems.append("the file contains no images")

    boxes: dict[int, list[GroundTruthBox]] = {image_id: [] for image_id in images}
    dropped = 0
    warnings: list[str] = []
    for i, ann in enumerate(data["annotations"]):
        if not isinstance(ann, dict):
            problems.append(f"annotations[{i}] is not an object")
            continue
        image_id, category_id, bbox = ann.get("image_id"), ann.get("category_id"), ann.get("bbox")
        if image_id not in images:
            problems.append(f"annotations[{i}] refers to unknown image_id {image_id!r}")
            continue
        if category_id not in category_to_class:
            problems.append(f"annotations[{i}] refers to unknown category_id {category_id!r}")
            continue
        if not (isinstance(bbox, list) and len(bbox) == 4 and all(_number(v) for v in bbox)):
            problems.append(f"annotations[{i}] needs 'bbox' = [x, y, width, height] (four finite numbers)")
            continue
        x, y, w, h = (float(v) for v in bbox)
        if w <= 0 or h <= 0:
            problems.append(f"annotations[{i}] has a box with width {w} and height {h}; both must be > 0")
            continue
        crowd = ann.get("iscrowd", 0)
        if crowd not in (0, 1, False, True):
            problems.append(f"annotations[{i}] has iscrowd {crowd!r}; expected 0 or 1")
            continue
        image = images[image_id]
        if x < 0 or y < 0 or x + w > image.width + 1 or y + h > image.height + 1:
            warnings.append(f"annotations[{i}] extends outside image {image.file_name} ({image.width} x {image.height})")
        class_id = category_to_class[category_id]
        if class_id == -1:
            dropped += 1
            continue
        boxes[image_id].append(GroundTruthBox((x, y, x + w, y + h), class_id, bool(crowd)))

    if problems:
        shown = "\n".join(f"  - {p}" for p in problems[:20])
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise AnnotationError(f"Invalid annotation file {path} ({len(problems)} problem(s)):\n{shown}{more}")
    present = sorted({c for c in category_to_class.values() if c != -1})
    if unmapped:
        warnings.insert(0, f"{len(unmapped)} category name(s) are not classes of this model and are not evaluated: "
                           f"{', '.join(unmapped)} ({dropped} annotation(s) dropped)")
    return GroundTruthDataset(
        images=tuple(images.values()),
        boxes={k: tuple(v) for k, v in boxes.items()},
        class_names={c: model_class_names[c] for c in present},
        unmapped_categories=tuple(unmapped),
        dropped_annotations=dropped,
        warnings=tuple(warnings),
    )
