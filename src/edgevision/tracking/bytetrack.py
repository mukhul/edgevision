"""ByteTrack multi-object tracking, using Ultralytics' implementation.

Usage::

    from edgevision.tracking import ByteTrackTracker

    tracker = ByteTrackTracker()            # Ultralytics' built-in bytetrack.yaml
    tracker.reset()                         # before each new video
    for frame_index, frame in reader:
        detections = detector.detect(frame, frame_index)
        detections = tracker.update(detections, frame)   # same list, IDs attached

How it fits together:

* The detector runs once per frame, as usual. Its ``Detection`` records are
  handed to Ultralytics' ``BYTETracker`` through Ultralytics' public ``Boxes``
  class, the input format ``BYTETracker.update`` documents. No second model
  and no second inference are involved, and any detector backend works.
* ``BYTETracker`` returns, for each track it reports, the index of the input
  detection it matched. That index attaches the track ID to the *original*
  detection, so boxes and confidences stay exactly as the detector produced them.
* Ultralytics needs the ``lap`` package for matching and would pip-install
  it at runtime if missing; this module checks for it first and raises a clear
  error instead (install the ``tracking`` extra).

ByteTrack matches detections to existing tracks in two passes: confident
detections first (score >= ``track_high_thresh``), then weaker ones (above
``track_low_thresh``) to continue tracks that the first pass missed. A track
that goes unmatched is kept as "lost" for ``track_buffer`` frames so it can be
recovered, then removed. New tracks start from unmatched detections scoring at
least ``new_track_thresh``; after the first frame a new track is only reported
once it has been matched in a second frame.
"""

from __future__ import annotations

import numbers
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from edgevision.models.base import Detection, validate_bgr_frame
from edgevision.optional import import_optional
from edgevision.tracking.base import TrackedDetection, TrackingError

if TYPE_CHECKING:
    import numpy as np

#: Settings ByteTrack needs, with a check for each. Values come from a tracker YAML.
_FRACTION = "a number between 0 and 1"
REQUIRED_SETTINGS: dict[str, str] = {
    "track_high_thresh": _FRACTION,
    "track_low_thresh": _FRACTION,
    "new_track_thresh": _FRACTION,
    "match_thresh": _FRACTION,
    "track_buffer": "a positive integer (frames)",
    "fuse_score": "true or false",
}


def validate_bytetrack_settings(settings: Any, source: str) -> dict[str, Any]:
    """Check a ByteTrack configuration and return it as a plain dict."""
    if not isinstance(settings, dict):
        raise TrackingError(f"Tracker config {source} must be a mapping, got {type(settings).__name__}")
    if settings.get("tracker_type") != "bytetrack":
        raise TrackingError(
            f"Tracker config {source} has tracker_type {settings.get('tracker_type')!r}; "
            "only 'bytetrack' is supported"
        )
    problems = []
    for key, expected in REQUIRED_SETTINGS.items():
        if key not in settings:
            problems.append(f"missing '{key}'")
            continue
        value = settings[key]
        if key == "fuse_score":
            ok = isinstance(value, bool)
        elif key == "track_buffer":
            ok = isinstance(value, numbers.Integral) and not isinstance(value, bool) and value >= 1
        else:
            ok = isinstance(value, numbers.Real) and not isinstance(value, bool) and 0 <= value <= 1
        if not ok:
            problems.append(f"'{key}' must be {expected}, got {value!r}")
    if problems:
        raise TrackingError(f"Invalid tracker config {source}: " + "; ".join(problems))
    return dict(settings)


def attach_track_ids(
    detections: Sequence[Detection], matches: Iterable[tuple[int, int, int]]
) -> list[Detection]:
    """Attach track IDs to detections.

    Args:
        detections: This frame's detections, in the order given to the tracker.
        matches: ``(detection_index, track_id, class_id)`` for each reported track.

    Returns:
        The same detections in the same order; matched ones become
        :class:`TrackedDetection`.

    Raises:
        TrackingError: if the matches are inconsistent with the detections.
    """
    result: list[Detection] = list(detections)
    used_indices: set[int] = set()
    used_ids: set[int] = set()
    for index, track_id, class_id in matches:
        if not 0 <= index < len(detections):
            raise TrackingError(
                f"Tracker referred to detection {index}, but only {len(detections)} were given"
            )
        if index in used_indices or track_id in used_ids:
            raise TrackingError(f"Tracker matched detection {index} / track {track_id} more than once")
        if class_id != detections[index].class_id:
            raise TrackingError(
                f"Tracker reported class {class_id} for detection {index}, "
                f"which has class {detections[index].class_id}"
            )
        used_indices.add(index)
        used_ids.add(track_id)
        result[index] = TrackedDetection.from_detection(detections[index], track_id)
    return result


class ByteTrackTracker:
    """Implements the :class:`~edgevision.tracking.base.Tracker` protocol with Ultralytics' ByteTrack.

    Args:
        config: Path to a ByteTrack YAML file (same format as Ultralytics'
            ``bytetrack.yaml``). ``None`` uses Ultralytics' built-in defaults.

    Raises:
        MissingDependencyError: Ultralytics or ``lap`` is not installed.
        TrackingError: the configuration is invalid.
    """

    name = "bytetrack"

    def __init__(self, config: str | os.PathLike[str] | None = None) -> None:
        # Check `lap` before importing Ultralytics' tracker code, which would
        # otherwise try to pip-install it.
        import_optional("lap", display_name="lap", extra="tracking", purpose="run ByteTrack tracking")
        import_optional("ultralytics", display_name="Ultralytics", extra="tracking", purpose="run ByteTrack tracking")
        import numpy as np
        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace

        self.config_source, raw = self._read_config(config)
        self.settings = validate_bytetrack_settings(raw, self.config_source)
        self._np = np
        self._boxes = Boxes
        self._backend = BYTETracker(IterableSimpleNamespace(**self.settings))

    @staticmethod
    def _read_config(config: str | os.PathLike[str] | None) -> tuple[str, Any]:
        if config is None:
            from ultralytics.utils import YAML
            from ultralytics.utils.checks import check_yaml

            return "ultralytics:bytetrack.yaml", YAML.load(check_yaml("bytetrack.yaml"))
        path = Path(config)
        if not path.is_file():
            raise TrackingError(f"Tracker config not found: {path} (looked relative to {Path.cwd()})")
        try:
            return str(path), yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            raise TrackingError(f"Could not read tracker config {path}: {exc}") from exc

    def reset(self) -> None:
        """Forget all tracks and restart track IDs at 1 (call before each new video)."""
        self._backend.reset()

    def update(self, detections: Sequence[Detection], frame: np.ndarray) -> list[Detection]:
        """Track one frame. Returns the detections with track IDs attached where assigned."""
        validate_bgr_frame(frame)
        if not all(isinstance(d, Detection) for d in detections):
            raise TypeError("detections must contain only Detection records")
        np = self._np
        data = np.array(
            [[*d.bbox_xyxy, d.confidence, d.class_id] for d in detections], dtype=np.float32
        ).reshape(-1, 6)  # Boxes format: x1, y1, x2, y2, confidence, class
        try:
            rows = self._backend.update(self._boxes(data, frame.shape[:2]), frame)
        except Exception as exc:
            raise TrackingError(f"ByteTrack update failed: {type(exc).__name__}: {exc}") from exc
        # Each row: x1, y1, x2, y2, track_id, score, class, detection_index
        matches = [(int(row[7]), int(row[4]), int(row[6])) for row in rows]
        return attach_track_ids(detections, matches)

    def __repr__(self) -> str:
        return f"ByteTrackTracker(config={self.config_source!r})"
