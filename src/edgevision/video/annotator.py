"""Draw detections onto a video frame.

Usage::

    from edgevision.video.annotator import annotate_frame

    annotated = annotate_frame(frame, detections)  # a new array; `frame` is unchanged

For each :class:`~edgevision.models.base.Detection` this draws a rectangle
from its ``bbox_xyxy`` and a filled label with the class name and confidence,
for example ``cat 0.91``. Colours depend only on the class ID, so a class
keeps the same colour in every frame and every run.

This module only draws: it knows nothing about models, inference or video
files. It needs OpenCV (the ``vision`` extra), imported when first used.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from edgevision.models.base import Detection, validate_bgr_frame
from edgevision.optional import import_optional

if TYPE_CHECKING:
    import numpy as np

#: 20 clearly distinct colours in BGR order (OpenCV's channel order).
PALETTE: tuple[tuple[int, int, int], ...] = (
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
    (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0),
    (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100), (236, 24, 0),
    (255, 56, 132), (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255),
)


def class_color(class_id: int) -> tuple[int, int, int]:
    """Deterministic BGR colour for a class ID (the palette repeats after 20 classes)."""
    return PALETTE[class_id % len(PALETTE)]


def _text_color(background: tuple[int, int, int]) -> tuple[int, int, int]:
    """Black or white, whichever is easier to read on ``background`` (BGR)."""
    blue, green, red = background
    luminance = 0.114 * blue + 0.587 * green + 0.299 * red
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def clip_box(
    bbox_xyxy: Sequence[float], width: int, height: int
) -> tuple[int, int, int, int] | None:
    """Round a box to whole pixels and clip it to the image.

    Returns ``None`` if the box lies completely outside the image. Pixel
    coordinates run from 0 to ``width - 1`` and 0 to ``height - 1``.
    """
    x1, y1, x2, y2 = bbox_xyxy
    if x2 < 0 or y2 < 0 or x1 > width - 1 or y1 > height - 1:
        return None
    return (
        min(max(round(x1), 0), width - 1),
        min(max(round(y1), 0), height - 1),
        min(max(round(x2), 0), width - 1),
        min(max(round(y2), 0), height - 1),
    )


def label_text(detection: Detection) -> str:
    return f"{detection.class_name} {detection.confidence:.2f}"


def annotate_frame(frame: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    """Return a copy of ``frame`` with every detection drawn on it.

    Args:
        frame: BGR ``uint8`` array of shape ``(height, width, 3)``; not modified.
        detections: Detections whose boxes are in this frame's pixel coordinates.

    Raises:
        ValueError: ``frame`` is not a valid BGR image.
        TypeError: an item in ``detections`` is not a ``Detection``.
    """
    validate_bgr_frame(frame)
    if not all(isinstance(d, Detection) for d in detections):
        raise TypeError("detections must contain only Detection records")
    cv2 = import_optional("cv2", display_name="OpenCV", extra="vision", purpose="draw annotations")

    canvas = frame.copy()  # never draw on the caller's frame
    height, width = canvas.shape[:2]
    # Scale lines and text with the image so labels stay readable at any size.
    thickness = max(1, round(min(width, height) / 300))
    font_scale = max(0.4, min(width, height) / 900)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for detection in detections:
        box = clip_box(detection.bbox_xyxy, width, height)
        if box is None:
            continue  # nothing of this box is visible
        x1, y1, x2, y2 = box
        color = class_color(detection.class_id)
        # Solid (non-anti-aliased) outline: crisp, and exactly the class colour.
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_8)

        text = label_text(detection)
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        label_h = text_h + baseline + 2 * thickness
        # Put the label above the box; if there is no room, put it just inside the top edge.
        top = y1 - label_h if y1 - label_h >= 0 else y1
        left = min(x1, max(0, width - text_w - 2 * thickness))  # keep it inside the frame
        cv2.rectangle(
            canvas, (left, top), (left + text_w + 2 * thickness, top + label_h), color, cv2.FILLED
        )
        cv2.putText(
            canvas, text, (left + thickness, top + text_h + thickness), font, font_scale,
            _text_color(color), thickness, cv2.LINE_AA,
        )
    return canvas
