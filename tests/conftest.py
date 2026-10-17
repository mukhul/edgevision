"""Shared test fixtures.

``fake_yolo`` replaces PyTorch and Ultralytics with tiny fakes, so detector
tests run without those packages, without model weights, without a GPU, and
without network access. A test configures what the fake "model" returns
through the controller object the fixture yields.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

FAKE_CLASS_NAMES = {0: "person", 5: "bus", 15: "cat"}


class FakeTensor:
    """Stands in for a torch.Tensor; the detector only calls .tolist()."""

    def __init__(self, values: list[Any]) -> None:
        self._values = values

    def tolist(self) -> list[Any]:
        return self._values


@dataclass
class FakeYoloController:
    """What the fake model does. Tests change these fields before creating a detector."""

    cuda_available: bool = False
    task: str = "detect"
    names: dict[int, str] = field(default_factory=lambda: dict(FAKE_CLASS_NAMES))
    # Each box: (x1, y1, x2, y2, confidence, class_id)
    boxes: list[tuple[float, float, float, float, float, float]] = field(default_factory=list)
    boxes_is_none: bool = False
    results_per_call: int = 1
    load_error: Exception | None = None
    predict_error: Exception | None = None
    # Recorded activity
    loaded_paths: list[str] = field(default_factory=list)
    predict_calls: list[dict[str, Any]] = field(default_factory=list)
    empty_cache_calls: int = 0

    def make_result(self) -> SimpleNamespace:
        if self.boxes_is_none:
            return SimpleNamespace(boxes=None)
        return SimpleNamespace(
            boxes=SimpleNamespace(
                xyxy=FakeTensor([list(b[:4]) for b in self.boxes]),
                conf=FakeTensor([b[4] for b in self.boxes]),
                cls=FakeTensor([float(b[5]) for b in self.boxes]),
            )
        )


@pytest.fixture
def fake_yolo(monkeypatch: pytest.MonkeyPatch) -> FakeYoloController:
    ctl = FakeYoloController()

    class FakeYOLO:
        def __init__(self, path: str) -> None:
            ctl.loaded_paths.append(path)
            if ctl.load_error is not None:
                raise ctl.load_error
            self.task = ctl.task
            self.names = ctl.names

        def predict(self, **kwargs: Any) -> list[SimpleNamespace]:
            ctl.predict_calls.append(kwargs)
            if ctl.predict_error is not None:
                raise ctl.predict_error
            return [ctl.make_result() for _ in range(ctl.results_per_call)]

    def empty_cache() -> None:
        ctl.empty_cache_calls += 1

    fake_torch = SimpleNamespace(
        __version__="2.14.1+cpu",
        version=SimpleNamespace(cuda=None),
        cuda=SimpleNamespace(is_available=lambda: ctl.cuda_available, empty_cache=empty_cache),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    return ctl


class FakeTracker:
    """Deterministic stand-in for ByteTrackTracker.

    Gives every class name its own ID (in order of first appearance), so the
    same class keeps the same ID across frames, and IDs restart at 1 after
    reset(). Detections whose class is in ``skip`` are left without an ID,
    like detections a real tracker has not confirmed yet.
    """

    name = "fake"

    def __init__(self, config: Any = None) -> None:
        self.config_source = "fake-config" if config is None else str(config)
        self.settings = {"tracker_type": "fake"}
        self.skip: set[str] = set()
        self.fail_at_frame: int | None = None
        self.fail_on_reset = False
        self.reset_calls = 0
        self.update_calls: list[int] = []
        self._ids: dict[str, int] = {}

    def reset(self) -> None:
        if self.fail_on_reset:
            raise RuntimeError("reset exploded")
        self.reset_calls += 1
        self._ids = {}

    def update(self, detections: list, frame: Any) -> list:
        from edgevision.tracking.base import TrackedDetection

        index = detections[0].frame_index if detections else None
        self.update_calls.append(index)
        if index is not None and index == self.fail_at_frame:
            raise RuntimeError("association blew up")
        result = []
        for d in detections:
            if d.class_name in self.skip:
                result.append(d)
                continue
            track_id = self._ids.setdefault(d.class_name, len(self._ids) + 1)
            result.append(TrackedDetection.from_detection(d, track_id))
        return result


@pytest.fixture
def fake_tracker_class() -> type[FakeTracker]:
    return FakeTracker
