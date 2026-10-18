"""Run timed benchmarks of a detector.

Two scopes, never mixed in one number:

``detector``
    Frames are decoded into memory **before** timing starts (an image is
    reused; a video contributes its first ``preload_frames`` frames, used in
    rotation). Each iteration times one ``detector.detect(frame)`` call: from a
    BGR frame in memory to a list of ``Detection`` records, i.e. preprocessing,
    inference and postprocessing. Backends that report stage times
    (``detector.last_timings``) also give preprocess / inference / postprocess
    samples for the same calls.

``pipeline`` (video only)
    Each iteration reads the next frame from the video file and runs the
    detector (and the tracker, if enabled). Decoding, detection and tracking
    are timed separately, plus their sum per frame. Drawing, video writing and
    JSON output are not part of any benchmark.

For every run a fresh detector is created (timed as initialisation), then
``warmup`` iterations run untimed for the statistics (their total time and
the very first call are recorded separately), then ``iterations`` measured
iterations. Throughput is measured iterations divided by the wall-clock time
of the measured loop.

All times come from ``time.perf_counter()`` (monotonic, high resolution).
Inference on the CPU is synchronous, and both backends return results as
CPU data (NumPy arrays / Python lists), so a ``detect()`` call cannot return
before the work is finished: no extra synchronisation is needed.
"""

from __future__ import annotations

import importlib
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any

from edgevision.benchmarks.stats import summarize, throughput

SCOPES = ("detector", "pipeline")
STAGES = ("preprocess_ms", "inference_ms", "postprocess_ms")
BACKEND_LIBRARIES = {"pytorch": ("torch", "ultralytics"), "onnx": ("onnxruntime",)}

Clock = Callable[[], float]


class BenchmarkError(Exception):
    """The benchmark could not be run as requested."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class BenchmarkPlan:
    """What to measure. Validated on creation."""

    scope: str = "detector"
    warmup: int = 20
    iterations: int = 300
    runs: int = 3
    preload_frames: int = 32
    tracking: bool = False

    def __post_init__(self) -> None:
        if self.scope not in SCOPES:
            raise ValueError(f"scope must be one of {list(SCOPES)}, got {self.scope!r}")
        if not _is_int(self.warmup) or self.warmup < 0:
            raise ValueError(f"warmup must be an integer >= 0, got {self.warmup!r}")
        for name in ("iterations", "runs", "preload_frames"):
            value = getattr(self, name)
            if not _is_int(value) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
        if not isinstance(self.tracking, bool):
            raise ValueError(f"tracking must be true or false, got {self.tracking!r}")
        if self.tracking and self.scope != "pipeline":
            raise ValueError("tracking can only be benchmarked in the pipeline scope")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "warmup_iterations": self.warmup,
            "measured_iterations": self.iterations,
            "runs": self.runs,
            "preload_frames": self.preload_frames if self.scope == "detector" else None,
            "tracking": self.tracking,
            "batch_size": 1,
        }


@dataclass
class RunResult:
    """Measurements of one run (one freshly created detector)."""

    run: int
    init_s: float
    first_inference_ms: float
    warmup_iterations: int
    warmup_s: float
    measured_iterations: int
    measured_wall_s: float
    samples_ms: dict[str, list[float]]
    detections_total: int
    device: str
    execution_providers: list[str] | None
    notes: list[str] = field(default_factory=list)
    tracker_init_s: float | None = None

    @property
    def throughput_fps(self) -> float:
        return throughput(self.measured_iterations, self.measured_wall_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": self.run,
            "init_s": self.init_s,
            "tracker_init_s": self.tracker_init_s,
            "first_inference_ms": self.first_inference_ms,
            "warmup_iterations": self.warmup_iterations,
            "warmup_s": self.warmup_s,
            "measured_iterations": self.measured_iterations,
            "measured_wall_s": self.measured_wall_s,
            "throughput_fps": self.throughput_fps,
            "detections_total": self.detections_total,
            "device": self.device,
            "execution_providers": self.execution_providers,
            "latency_ms": {name: summarize(values) for name, values in self.samples_ms.items()},
            "samples_ms": self.samples_ms,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Library import time
# ---------------------------------------------------------------------------


def time_library_imports(backend: str, clock: Clock = time.perf_counter) -> dict[str, Any]:
    """Import the backend's main libraries and time it, before any model is loaded.

    Measured once per benchmark process. If a library was already imported
    (``already_imported``), the time is close to zero and says nothing about
    a cold start. Missing libraries are reported, not raised: creating the
    detector then fails with the usual install hint.
    """
    modules = BACKEND_LIBRARIES[backend]
    already = all(name in sys.modules for name in modules)
    start = clock()
    try:
        for name in modules:
            importlib.import_module(name)
    except ImportError as exc:
        return {"modules": list(modules), "seconds": None, "already_imported": already, "error": str(exc)}
    return {"modules": list(modules), "seconds": clock() - start, "already_imported": already, "error": None}


# ---------------------------------------------------------------------------
# One run of each scope
# ---------------------------------------------------------------------------


def _create(make_detector: Callable[[], Any], clock: Clock) -> tuple[Any, float]:
    start = clock()
    detector = make_detector()
    return detector, clock() - start


def _stage_samples(collected: list[dict[str, float] | None]) -> tuple[dict[str, list[float]], str | None]:
    """Stage samples if every measured call reported them, else none with a reason."""
    if not collected or any(t is None for t in collected):
        return {}, "this backend did not report stage timings for every measured call"
    try:
        return {stage: [float(t[stage]) for t in collected] for stage in STAGES}, None
    except (KeyError, TypeError, ValueError):
        return {}, "this backend's stage timings were incomplete"


def _device_info(detector: Any) -> tuple[str, list[str] | None]:
    providers = getattr(detector, "providers", None)
    return str(getattr(detector, "device", "unknown")), (list(providers) if providers is not None else None)


def run_detector_once(
    make_detector: Callable[[], Any],
    frames: Sequence[Any],
    plan: BenchmarkPlan,
    run: int,
    clock: Clock = time.perf_counter,
) -> RunResult:
    """One run of the ``detector`` scope on frames that are already in memory."""
    if not frames:
        raise BenchmarkError("no frames to benchmark")
    detector, init_s = _create(make_detector, clock)
    notes: list[str] = []
    try:
        first_ms = None
        warmup_start = clock()
        for i in range(plan.warmup):
            start = clock()
            detector.detect(frames[i % len(frames)], frame_index=i)
            if i == 0:
                first_ms = (clock() - start) * 1e3
        warmup_s = clock() - warmup_start

        detect_ms: list[float] = []
        stage_timings: list[dict[str, float] | None] = []
        detections = 0
        loop_start = clock()
        for i in range(plan.iterations):
            frame = frames[i % len(frames)]
            start = clock()
            result = detector.detect(frame, frame_index=i)
            detect_ms.append((clock() - start) * 1e3)
            stage_timings.append(getattr(detector, "last_timings", None))
            detections += len(result)
        loop_s = clock() - loop_start
        device, providers = _device_info(detector)
    finally:
        detector.close()

    if first_ms is None:
        first_ms = detect_ms[0]
        notes.append("warm-up was 0: the first measured iteration includes one-off setup costs")
    stages, missing = _stage_samples(stage_timings)
    if missing:
        notes.append(f"stage breakdown unavailable: {missing}")
    return RunResult(
        run=run, init_s=init_s, first_inference_ms=first_ms, warmup_iterations=plan.warmup, warmup_s=warmup_s,
        measured_iterations=plan.iterations, measured_wall_s=loop_s, samples_ms={"detect": detect_ms, **stages},
        detections_total=detections, device=device, execution_providers=providers, notes=notes,
    )


def run_pipeline_once(
    make_detector: Callable[[], Any],
    open_video: Callable[[], AbstractContextManager[Iterable[tuple[int, Any]]]],
    plan: BenchmarkPlan,
    run: int,
    make_tracker: Callable[[], Any] | None = None,
    clock: Clock = time.perf_counter,
) -> RunResult:
    """One run of the ``pipeline`` scope: decode + detect (+ track) per frame, from the start of the video."""
    detector, init_s = _create(make_detector, clock)
    tracker, tracker_init_s = (None, None)
    notes: list[str] = []
    try:
        if plan.tracking:
            if make_tracker is None:
                raise BenchmarkError("tracking was requested but no tracker factory was given")
            tracker, tracker_init_s = _create(make_tracker, clock)
            tracker.reset()
        with open_video() as video:
            frames = iter(video)

            def step() -> tuple[float, float, float, float, int, Any] | None:
                t0 = clock()
                item = next(frames, None)  # decoding
                t1 = clock()
                if item is None:
                    return None
                index, frame = item
                detections = detector.detect(frame, frame_index=index)
                t2 = clock()
                if tracker is not None:
                    detections = tracker.update(detections, frame)
                t3 = clock()
                timings = getattr(detector, "last_timings", None)
                return t0, t1, t2, t3, len(detections), timings

            first_ms = None
            warmup_start = clock()
            for i in range(plan.warmup):
                step_result = step()
                if step_result is None:
                    raise BenchmarkError(
                        f"the video ended after {i} frames, during the {plan.warmup} warm-up frames; "
                        "use a longer video or fewer warm-up iterations"
                    )
                if i == 0:
                    first_ms = (step_result[2] - step_result[1]) * 1e3
            warmup_s = clock() - warmup_start

            samples: dict[str, list[float]] = {"decode": [], "detect": []}
            if tracker is not None:
                samples["track"] = []
            samples["frame"] = []
            stage_timings: list[dict[str, float] | None] = []
            detections = 0
            loop_start = last_end = clock()
            for _ in range(plan.iterations):
                step_result = step()
                if step_result is None:
                    break
                t0, t1, t2, t3, count, timings = step_result
                samples["decode"].append((t1 - t0) * 1e3)
                samples["detect"].append((t2 - t1) * 1e3)
                if tracker is not None:
                    samples["track"].append((t3 - t2) * 1e3)
                samples["frame"].append((t3 - t0) * 1e3)
                stage_timings.append(timings)
                detections += count
                last_end = t3
        device, providers = _device_info(detector)
    finally:
        detector.close()

    measured = len(samples["frame"])
    if measured == 0:
        raise BenchmarkError(
            f"the video has no frames left after the {plan.warmup} warm-up frames; use a longer video or less warm-up"
        )
    if measured < plan.iterations:
        notes.append(f"the video ended after {measured} of {plan.iterations} measured frames")
    if first_ms is None:
        first_ms = samples["detect"][0]
        notes.append("warm-up was 0: the first measured iteration includes one-off setup costs")
    stages, missing = _stage_samples(stage_timings)
    if missing:
        notes.append(f"stage breakdown unavailable: {missing}")
    return RunResult(
        run=run, init_s=init_s, first_inference_ms=first_ms, warmup_iterations=plan.warmup, warmup_s=warmup_s,
        measured_iterations=measured, measured_wall_s=last_end - loop_start, samples_ms={**samples, **stages},
        detections_total=detections, device=device, execution_providers=providers, notes=notes,
        tracker_init_s=tracker_init_s,
    )


def run_benchmark(
    plan: BenchmarkPlan,
    make_detector: Callable[[], Any],
    *,
    frames: Sequence[Any] | None = None,
    open_video: Callable[[], Any] | None = None,
    make_tracker: Callable[[], Any] | None = None,
    clock: Clock = time.perf_counter,
    progress: Callable[[str], None] | None = None,
) -> list[RunResult]:
    """All runs of a plan. ``frames`` is needed for the detector scope, ``open_video`` for the pipeline scope."""
    results = []
    for run in range(1, plan.runs + 1):
        if progress is not None:
            progress(f"run {run}/{plan.runs}: {plan.warmup} warm-up + {plan.iterations} measured iterations")
        if plan.scope == "detector":
            if frames is None:
                raise BenchmarkError("the detector scope needs frames in memory")
            results.append(run_detector_once(make_detector, frames, plan, run, clock))
        else:
            if open_video is None:
                raise BenchmarkError("the pipeline scope needs a video")
            results.append(run_pipeline_once(make_detector, open_video, plan, run, make_tracker, clock))
    return results


# ---------------------------------------------------------------------------
# Input shape actually fed to the model
# ---------------------------------------------------------------------------


def model_input_shape(height: int, width: int, input_size: int, letterbox: str, stride: int = 32) -> list[int]:
    """``[1, 3, H, W]`` of the tensor the network receives for a ``height`` x ``width`` frame.

    ``square`` pads to ``input_size`` x ``input_size``. ``minimal`` (Ultralytics'
    default for ``.pt`` models) pads only to the next multiple of ``stride``.
    """
    if letterbox == "square":
        return [1, 3, input_size, input_size]
    if letterbox != "minimal":
        raise ValueError(f"letterbox must be 'square' or 'minimal', got {letterbox!r}")
    ratio = min(input_size / height, input_size / width)
    new_w, new_h = round(width * ratio), round(height * ratio)
    pad_w, pad_h = (input_size - new_w) % stride, (input_size - new_h) % stride
    return [1, 3, new_h + pad_h, new_w + pad_w]
