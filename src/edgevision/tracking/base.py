"""Backend-independent tracking types.

A **detection** says "there is a cat here in this frame". A **tracked
detection** adds "and it is the same cat as track #3 in earlier frames". The
tracker supplies that link; the detector knows nothing about it, and
``Detection`` itself stays free of tracking state.

Tracker contract (see :class:`Tracker`): ``update(detections, frame)`` returns
the *same* detections, in the *same order*, with each one the tracker
assigned to a track replaced by a :class:`TrackedDetection`. Detections the
tracker did not (yet) assign stay plain ``Detection`` records. So enabling
tracking never adds, drops or reorders detections.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from edgevision.models.base import Detection, _require_non_negative_int

if TYPE_CHECKING:
    import numpy as np


class TrackingError(Exception):
    """The tracker failed or returned inconsistent results."""


@dataclass(frozen=True)
class TrackedDetection(Detection):
    """A detection plus the ID of the track it belongs to.

    ``track_id`` identifies a track *within one video run*. It is not a
    persistent identity: if an object is lost (occlusion, missed detections)
    for long enough, it may come back with a new ID.
    """

    track_id: int

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "track_id", _require_non_negative_int("track_id", self.track_id))

    @classmethod
    def from_detection(cls, detection: Detection, track_id: int) -> TrackedDetection:
        return cls(
            detection.frame_index,
            detection.bbox_xyxy,
            detection.confidence,
            detection.class_id,
            detection.class_name,
            track_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "track_id": self.track_id}


@runtime_checkable
class Tracker(Protocol):
    """What every tracker provides.

    ``update`` is called once per frame, in frame order, with that frame's
    detections. ``reset`` forgets all tracks and restarts IDs; it must be
    called before each new video so identities never leak between videos.
    """

    name: str

    def update(self, detections: Sequence[Detection], frame: np.ndarray) -> list[Detection]: ...

    def reset(self) -> None: ...
