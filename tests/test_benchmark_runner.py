"""Tests for edgevision.benchmarks.runner.

A fake clock and fake detectors make every timing exact: each fake call
advances the clock by a programmed amount, so the tests can check precisely
which intervals end up in which metric. No real model or timing is involved,
and none of these numbers are performance measurements.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager

import pytest

from edgevision.benchmarks.runner import (
    BenchmarkError,
    BenchmarkPlan,
    model_input_shape,
    run_benchmark,
    run_detector_once,
    run_pipeline_once,
    time_library_imports,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeDetector:
    """detect() takes ``durations[i]`` seconds on the fake clock and returns ``per_frame`` detections."""

    created: list[FakeDetector] = []

    def __init__(self, clock: FakeClock, durations: list[float], per_frame: int = 2, stages: bool = True) -> None:
        self.clock, self.durations, self.per_frame, self.stages = clock, list(durations), per_frame, stages
        self.calls: list[tuple[str, int]] = []
        self.closed = False
        self.last_timings = None
        self.device = "cpu"
        self.providers = ("CPUExecutionProvider",)

    def detect(self, frame, frame_index=0):
        assert not self.closed
        duration = self.durations.pop(0)
        self.clock.advance(duration)
        self.calls.append((frame, frame_index))
        if self.stages:
            ms = duration * 1e3
            self.last_timings = {"preprocess_ms": ms * 0.1, "inference_ms": ms * 0.8, "postprocess_ms": ms * 0.1}
        return ["d"] * self.per_frame

    def close(self) -> None:
        self.closed = True


def factory(clock: FakeClock, durations: list[float], init_s: float = 2.0, **kw):
    created = []

    def make():
        clock.advance(init_s)
        detector = FakeDetector(clock, durations, **kw)
        created.append(detector)
        return detector

    make.created = created
    return make


# --- plan validation ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scope": "model_only"}, "scope must be one of"),
        ({"warmup": -1}, "warmup must be an integer >= 0"),
        ({"warmup": True}, "warmup must be an integer >= 0"),
        ({"iterations": 0}, "iterations must be an integer >= 1"),
        ({"iterations": 2.5}, "iterations must be an integer >= 1"),
        ({"runs": 0}, "runs must be an integer >= 1"),
        ({"preload_frames": 0}, "preload_frames must be an integer >= 1"),
        ({"tracking": True}, "only be benchmarked in the pipeline scope"),
        ({"scope": "pipeline", "tracking": "yes"}, "tracking must be true or false"),
    ],
)
def test_invalid_plans(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        BenchmarkPlan(**kwargs)


def test_plan_to_dict() -> None:
    plan = BenchmarkPlan(scope="pipeline", warmup=0, iterations=5, runs=1, tracking=True)
    assert plan.to_dict() == {"scope": "pipeline", "warmup_iterations": 0, "measured_iterations": 5, "runs": 1,
                              "preload_frames": None, "tracking": True, "batch_size": 1}


# --- detector scope ------------------------------------------------------------------------------


def test_warmup_and_initialisation_are_excluded_from_latency() -> None:
    clock = FakeClock()
    make = factory(clock, [5.0, 1.0, 1.0, 0.010, 0.020, 0.030], init_s=2.0)
    result = run_detector_once(make, ["f0", "f1"], BenchmarkPlan(warmup=3, iterations=3, runs=1), run=1, clock=clock)

    assert result.init_s == pytest.approx(2.0)
    assert result.first_inference_ms == pytest.approx(5000.0)  # the first warm-up call, reported separately
    assert result.warmup_s == pytest.approx(7.0)
    assert result.samples_ms["detect"] == pytest.approx([10.0, 20.0, 30.0])  # only the measured calls
    assert result.samples_ms["inference_ms"] == pytest.approx([8.0, 16.0, 24.0])
    assert result.measured_wall_s == pytest.approx(0.060)
    assert result.throughput_fps == pytest.approx(50.0)  # 3 frames / 0.06 s
    assert result.detections_total == 6
    assert result.device == "cpu" and result.execution_providers == ["CPUExecutionProvider"]
    assert result.notes == []


def test_frames_are_used_in_rotation_and_the_detector_is_reused() -> None:
    clock = FakeClock()
    make = factory(clock, [0.01] * 7)
    run_detector_once(make, ["a", "b", "c"], BenchmarkPlan(warmup=2, iterations=5, runs=1), run=1, clock=clock)
    detector = make.created[0]
    assert len(make.created) == 1  # one detector for the whole run
    assert [f for f, _ in detector.calls] == ["a", "b", "a", "b", "c", "a", "b"]  # warm-up, then measured from frame 0
    assert [i for _, i in detector.calls[2:]] == [0, 1, 2, 3, 4]
    assert detector.closed


def test_zero_warmup_is_allowed_but_noted() -> None:
    clock = FakeClock()
    result = run_detector_once(factory(clock, [0.5, 0.01]), ["f"], BenchmarkPlan(warmup=0, iterations=2), 1, clock)
    assert result.first_inference_ms == pytest.approx(500.0)
    assert result.samples_ms["detect"] == pytest.approx([500.0, 10.0])  # the slow first call is in the statistics
    assert any("warm-up was 0" in n for n in result.notes)


def test_missing_stage_timings_are_reported_as_unavailable() -> None:
    clock = FakeClock()
    result = run_detector_once(factory(clock, [0.01] * 3, stages=False), ["f"], BenchmarkPlan(warmup=1, iterations=2),
                               1, clock)
    assert set(result.samples_ms) == {"detect"}  # no invented stage numbers
    assert any("stage breakdown unavailable" in n for n in result.notes)


def test_detector_is_closed_when_a_call_fails() -> None:
    clock = FakeClock()
    make = factory(clock, [0.01])  # only one duration: the second call fails
    with pytest.raises(IndexError):
        run_detector_once(make, ["f"], BenchmarkPlan(warmup=1, iterations=2), 1, clock)
    assert make.created[0].closed


def test_no_frames() -> None:
    with pytest.raises(BenchmarkError, match="no frames"):
        run_detector_once(factory(FakeClock(), []), [], BenchmarkPlan(), 1)


def test_every_run_gets_a_new_detector() -> None:
    clock = FakeClock()
    make = factory(clock, [0.01] * 3)
    messages = []
    results = run_benchmark(BenchmarkPlan(warmup=1, iterations=2, runs=3), make, frames=["f"], clock=clock,
                            progress=messages.append)
    assert [r.run for r in results] == [1, 2, 3]
    assert len(make.created) == 3 and all(d.closed for d in make.created)
    assert len(messages) == 3 and "run 1/3" in messages[0]


def test_run_benchmark_needs_the_right_input() -> None:
    with pytest.raises(BenchmarkError, match="needs frames"):
        run_benchmark(BenchmarkPlan(runs=1), factory(FakeClock(), []))
    with pytest.raises(BenchmarkError, match="needs a video"):
        run_benchmark(BenchmarkPlan(scope="pipeline", runs=1), factory(FakeClock(), []))


# --- pipeline scope ------------------------------------------------------------------------------


def fake_video(clock: FakeClock, frames: int, decode_s: float = 0.004):
    opened = []

    @contextmanager
    def open_video():
        def frames_iter():
            for i in range(frames):
                clock.advance(decode_s)
                yield i, f"frame{i}"
            clock.advance(decode_s)  # the failed read at the end of the video also takes time

        opened.append(True)
        yield frames_iter()

    open_video.opened = opened
    return open_video


class FakeTracker:
    def __init__(self, clock: FakeClock) -> None:
        self.clock, self.resets, self.updates = clock, 0, 0

    def reset(self) -> None:
        self.resets += 1

    def update(self, detections, frame):
        self.clock.advance(0.001)
        self.updates += 1
        return detections


def test_pipeline_times_decode_detect_and_track_separately() -> None:
    clock = FakeClock()
    trackers = []

    def make_tracker():
        clock.advance(0.5)
        trackers.append(FakeTracker(clock))
        return trackers[-1]

    plan = BenchmarkPlan(scope="pipeline", warmup=2, iterations=3, runs=1, tracking=True)
    result = run_pipeline_once(factory(clock, [1.0, 0.02, 0.010, 0.010, 0.010]), fake_video(clock, 10), plan, 1,
                               make_tracker, clock)

    assert result.tracker_init_s == pytest.approx(0.5)
    assert result.first_inference_ms == pytest.approx(1000.0)  # detection only, not decoding
    assert result.samples_ms["decode"] == pytest.approx([4.0] * 3)
    assert result.samples_ms["detect"] == pytest.approx([10.0] * 3)
    assert result.samples_ms["track"] == pytest.approx([1.0] * 3)
    assert result.samples_ms["frame"] == pytest.approx([15.0] * 3)  # decode + detect + track
    assert list(result.samples_ms)[:4] == ["decode", "detect", "track", "frame"]
    assert result.measured_wall_s == pytest.approx(0.045)
    assert result.throughput_fps == pytest.approx(3 / 0.045)
    assert trackers[0].resets == 1 and trackers[0].updates == 5


def test_pipeline_video_ends_early() -> None:
    clock = FakeClock()
    plan = BenchmarkPlan(scope="pipeline", warmup=1, iterations=10, runs=1)
    result = run_pipeline_once(factory(clock, [0.01] * 4), fake_video(clock, 4), plan, 1, clock=clock)
    assert result.measured_iterations == 3
    assert result.measured_wall_s == pytest.approx(3 * 0.014)  # ends at the last frame, not at the failed read
    assert any("ended after 3 of 10" in n for n in result.notes)
    assert "track" not in result.samples_ms


def test_pipeline_video_too_short() -> None:
    clock = FakeClock()
    with pytest.raises(BenchmarkError, match="during the 5 warm-up frames"):
        run_pipeline_once(factory(clock, [0.01] * 9), fake_video(clock, 3),
                          BenchmarkPlan(scope="pipeline", warmup=5, iterations=2), 1, clock=clock)
    make = factory(clock, [0.01] * 9)
    with pytest.raises(BenchmarkError, match="no frames left"):
        run_pipeline_once(make, fake_video(clock, 2), BenchmarkPlan(scope="pipeline", warmup=2, iterations=2), 1,
                          clock=clock)
    assert make.created[0].closed


def test_pipeline_each_run_restarts_the_video() -> None:
    clock = FakeClock()
    video = fake_video(clock, 5)
    results = run_benchmark(BenchmarkPlan(scope="pipeline", warmup=1, iterations=2, runs=2),
                            factory(clock, [0.01] * 3), open_video=video, clock=clock)
    assert len(video.opened) == 2 and all(r.measured_iterations == 2 for r in results)


def test_tracking_needs_a_tracker_factory() -> None:
    clock = FakeClock()
    with pytest.raises(BenchmarkError, match="no tracker factory"):
        run_pipeline_once(factory(clock, [0.01]), fake_video(clock, 3),
                          BenchmarkPlan(scope="pipeline", tracking=True), 1, clock=clock)


# --- helpers -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("height", "width", "letterbox", "expected"),
    [
        (576, 1024, "square", [1, 3, 640, 640]),
        (576, 1024, "minimal", [1, 3, 384, 640]),  # 360 rows padded to the next multiple of 32
        (1080, 810, "minimal", [1, 3, 640, 480]),
        (480, 640, "minimal", [1, 3, 480, 640]),
        (51, 100, "minimal", [1, 3, 352, 640]),
    ],
)
def test_model_input_shape(height: int, width: int, letterbox: str, expected: list[int]) -> None:
    assert model_input_shape(height, width, 640, letterbox) == expected


@pytest.mark.parametrize(("height", "width"), [(576, 1024), (1080, 810), (51, 100), (333, 17), (720, 1280)])
def test_minimal_shape_matches_ultralytics(height: int, width: int) -> None:
    np = pytest.importorskip("numpy")
    augment = pytest.importorskip("ultralytics.data.augment", reason="Ultralytics not installed")
    image = augment.LetterBox(640, auto=True, stride=32)(image=np.zeros((height, width, 3), np.uint8))
    assert model_input_shape(height, width, 640, "minimal") == [1, 3, image.shape[0], image.shape[1]]


def test_invalid_letterbox() -> None:
    with pytest.raises(ValueError, match="letterbox"):
        model_input_shape(10, 10, 640, "stretch")


def test_library_import_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", object())
    info = time_library_imports("onnx")
    assert info["already_imported"] is True and info["seconds"] >= 0 and info["error"] is None
    monkeypatch.setitem(sys.modules, "onnxruntime", None)  # import fails
    info = time_library_imports("onnx")
    assert info["seconds"] is None and info["error"]
