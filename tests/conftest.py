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


# --- fake ONNX Runtime ------------------------------------------------------------------

FAKE_ONNX_NAMES = {0: "person", 1: "bus", 2: "cat"}
CPU_EP, CUDA_EP = "CPUExecutionProvider", "CUDAExecutionProvider"


@dataclass
class FakeOrtController:
    """What the fake ONNX Runtime session looks like and returns.

    The default is a YOLO-style export with 3 classes: input ``images``
    float32 [1, 3, 640, 640] and output ``output0`` [1, 7, 8400]
    (4 box values + 3 class scores per candidate), with no NMS inside.
    """

    available: list[str] = field(default_factory=lambda: [CPU_EP])
    activated: list[str] | None = None  # providers the session reports; None = what was requested
    input_name: str = "images"
    input_shape: list[Any] = field(default_factory=lambda: [1, 3, 640, 640])
    input_type: str = "tensor(float)"
    outputs: list[tuple[str, list[Any], str]] = field(
        default_factory=lambda: [("output0", [1, 7, 8400], "tensor(float)")]
    )
    metadata: dict[str, str] = field(
        default_factory=lambda: {"task": "detect", "names": repr(FAKE_ONNX_NAMES), "args": "{'nms': None}"}
    )
    output: Any = None  # array returned by run(); None = zeros of the declared output shape
    load_error: Exception | None = None
    run_error: Exception | None = None
    # Recorded activity
    sessions: list[dict[str, Any]] = field(default_factory=list)
    feeds: list[dict[str, Any]] = field(default_factory=list)


@pytest.fixture
def fake_ort(monkeypatch: pytest.MonkeyPatch) -> FakeOrtController:
    import numpy as np

    ctl = FakeOrtController()

    class FakeSession:
        def __init__(self, path: str, sess_options: Any = None, providers: list[str] | None = None) -> None:
            ctl.sessions.append({"path": path, "providers": list(providers or [])})
            if ctl.load_error is not None:
                raise ctl.load_error
            self._providers = list(providers or []) if ctl.activated is None else list(ctl.activated)

        def get_providers(self) -> list[str]:
            return self._providers

        def get_inputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name=ctl.input_name, shape=ctl.input_shape, type=ctl.input_type)]

        def get_outputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name=n, shape=s, type=t) for n, s, t in ctl.outputs]

        def get_modelmeta(self) -> SimpleNamespace:
            return SimpleNamespace(custom_metadata_map=dict(ctl.metadata))

        def run(self, output_names: list[str], feeds: dict[str, Any]) -> list[Any]:
            ctl.feeds.append({"output_names": output_names, **feeds})
            if ctl.run_error is not None:
                raise ctl.run_error
            if ctl.output is not None:
                return [ctl.output]
            shape = [d if isinstance(d, int) else 1 for d in ctl.outputs[0][1]]
            return [np.zeros(shape, dtype=np.float32)]

    module = SimpleNamespace(
        __version__="1.31.0",
        get_available_providers=lambda: list(ctl.available),
        SessionOptions=lambda: SimpleNamespace(),
        InferenceSession=FakeSession,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", module)
    return ctl


def raw_yolo_output(candidates: list[tuple[float, float, float, float, int, float]], num_classes: int = 3,
                    total: int = 8400) -> Any:
    """Build a (1, 4 + classes, total) raw YOLO output from (cx, cy, w, h, class_id, score) rows.

    Unused candidate columns are zero (score 0), like empty anchors.
    """
    import numpy as np

    out = np.zeros((1, 4 + num_classes, total), dtype=np.float32)
    for i, (cx, cy, w, h, class_id, score) in enumerate(candidates):
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4 + class_id, i] = score
    return out


@pytest.fixture
def make_raw_output():
    """The :func:`raw_yolo_output` helper, as a fixture."""
    return raw_yolo_output
