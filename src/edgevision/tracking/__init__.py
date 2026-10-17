"""Multi-object tracking.

Importing this package does not require Ultralytics or ``lap``; the ByteTrack
adapter imports them only when a tracker is created.
"""

from edgevision.tracking.base import TrackedDetection, Tracker, TrackingError
from edgevision.tracking.bytetrack import ByteTrackTracker

__all__ = ["ByteTrackTracker", "TrackedDetection", "Tracker", "TrackingError"]
