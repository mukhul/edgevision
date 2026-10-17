"""Save video detection results as JSON, incrementally and safely.

Frames are written to disk as they are processed, so the full set of
detections never has to be held in memory. Everything is written to
``<name>.partial`` first and renamed to ``<name>`` only after the summary has
been written, so an interrupted or failed run never leaves a file that looks
complete. (A crash can leave the ``.partial`` file behind; it is incomplete
by definition and safe to delete.)

File layout (schema ``edgevision.video_detections``, version 1)::

    {
      "schema": "edgevision.video_detections",
      "schema_version": 1,
      "edgevision_version": "...",
      "created_utc": "2026-10-17T11:00:00Z",
      "video": {"path", "width", "height", "fps", "frame_count_reported", "fourcc"},
      "detector": {"backend", "model", "device", "settings": {...}},
      "bbox_format": "xyxy_pixels",
      "frames": [
        {"frame_index": 0, "detections": [
            {"bbox_xyxy": [x1, y1, x2, y2], "confidence": 0.91, "class_id": 15, "class_name": "cat"}
        ]},
        ...                                   # one entry per processed frame, in order
      ],
      "summary": {"frames_processed", "frames_with_detections", "frames_without_detections",
                  "total_detections", "detections_by_class", "max_frames", "completed",
                  "stop_reason", "ended_before_reported_count"}
    }

Unknown values (e.g. a video whose FPS is not reported) are ``null``.

With tracking enabled (an additive, backward-compatible extension of v1):
a top-level ``"tracker": {"name", "config", "settings"}`` block after
``bbox_format``; a ``"track_id"`` key on every detection (an integer, or
``null`` if the tracker did not assign one); and ``summary.tracking`` with
``tracker``, ``detections_with_track_id`` and ``distinct_track_ids_observed``.
Detection-only files contain none of these keys.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import IO, Any

import edgevision
from edgevision.pipeline import FrameDetections
from edgevision.video.reader import VideoMetadata

SCHEMA_NAME = "edgevision.video_detections"
SCHEMA_VERSION = 1
BBOX_FORMAT = "xyxy_pixels"  # (x1, y1, x2, y2) in pixels of the original frame


def _dumps(value: Any) -> str:
    # allow_nan=False: NaN/Infinity are not valid JSON, so fail loudly instead.
    return json.dumps(value, allow_nan=False)


def video_header(
    *,
    metadata: VideoMetadata,
    backend: str,
    model: str,
    device: str,
    settings: dict[str, Any],
    tracker: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The descriptive part of a results file, written before the frames.

    ``tracker`` (name and settings) is added only when tracking was used, so
    detection-only files keep exactly their original layout.
    """
    header = {
        "edgevision_version": edgevision.__version__,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "video": {
            "path": metadata.path.as_posix(),
            "width": metadata.width,
            "height": metadata.height,
            "fps": metadata.fps,
            "frame_count_reported": metadata.frame_count,
            "fourcc": metadata.fourcc,
        },
        "detector": {"backend": backend, "model": model, "device": device, "settings": settings},
        "bbox_format": BBOX_FORMAT,
    }
    if tracker is not None:
        header["tracker"] = tracker
    return header


class VideoDetectionsWriter:
    """Stream per-frame detections to a JSON file, then atomically publish it.

    Use as a context manager; call :meth:`finish` with the summary at the end.
    If the block exits without ``finish`` (an error, Ctrl+C), the partial
    file is deleted and no results file is created.

    Raises:
        FileExistsError: ``path`` exists and ``overwrite`` is False.
    """

    def __init__(self, path: str | os.PathLike[str], header: dict[str, Any], *, overwrite: bool = False) -> None:
        self.path = Path(path)
        if self.path.is_dir():
            raise IsADirectoryError(f"Results path is a directory: {self.path}")
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"{self.path} already exists; use --overwrite to replace it")
        self.partial_path = self.path.with_name(self.path.name + ".partial")
        self.frames_written = 0
        self._finished = False

        self.path.parent.mkdir(parents=True, exist_ok=True)
        fields = {"schema": SCHEMA_NAME, "schema_version": SCHEMA_VERSION, **header}
        self._file: IO[str] | None = self.partial_path.open("w", encoding="utf-8", newline="\n")
        try:
            self._file.write("{\n")
            for key, value in fields.items():
                self._file.write(f"  {_dumps(key)}: {_dumps(value)},\n")
            self._file.write('  "frames": [')
        except BaseException:
            self.abort()
            raise

    def write_frame(self, frame: FrameDetections) -> None:
        file = self._require_open()
        separator = "," if self.frames_written else ""
        file.write(f"{separator}\n    {_dumps(frame.to_dict())}")
        self.frames_written += 1

    def finish(self, summary: dict[str, Any]) -> Path:
        """Write the summary, close the file and move it to its final name."""
        file = self._require_open()
        file.write(f"\n  ],\n  \"summary\": {_dumps(summary)}\n}}\n")
        file.flush()
        os.fsync(file.fileno())  # make sure the bytes are on disk before the rename
        file.close()
        self._file = None
        os.replace(self.partial_path, self.path)  # atomic on the same filesystem
        self._finished = True
        return self.path

    def abort(self) -> None:
        """Discard everything written so far."""
        if self._file is not None:
            self._file.close()
            self._file = None
        self.partial_path.unlink(missing_ok=True)

    def _require_open(self) -> IO[str]:
        if self._file is None:
            raise ValueError("This results writer is already finished or aborted.")
        return self._file

    def __enter__(self) -> VideoDetectionsWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if not self._finished:
            self.abort()
