"""Benchmark result documents: metadata, JSON schema, summaries and comparison.

Schema ``edgevision.benchmark`` version 1 (documented in ``results/README.md``).
Times are in milliseconds (per-iteration latencies) or seconds (initialisation,
warm-up, measured interval), at full float precision. Paths are stored
relative to the working directory when possible, never as absolute paths that
would reveal user names; files are identified by size and SHA-256 instead.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import edgevision
from edgevision.benchmarks.runner import RunResult
from edgevision.benchmarks.stats import summarize

SCHEMA_NAME = "edgevision.benchmark"
SCHEMA_VERSION = 1

#: What each reported metric means. Stored in every result file.
TIMING_DEFINITIONS = {
    "clock": "time.perf_counter() (monotonic, high resolution). Latencies in milliseconds, other durations in seconds.",
    "library_import_s": "Importing the backend's main libraries (PyTorch + Ultralytics, or ONNX Runtime), once per "
                        "process before any model is loaded. Near zero if they were already imported.",
    "init_s": "Creating the detector: loading the model / creating the ONNX Runtime session and checking it. "
              "Measured for every run; later runs can be faster because files are cached by the operating system.",
    "first_inference_ms": "The first detect() call after initialisation. Often much slower than later calls "
                          "(lazy allocation, Ultralytics' internal warm-up). Not part of the statistics.",
    "warmup_s": "Total time of the warm-up iterations, which are excluded from all statistics.",
    "latency_ms.detect": "One detector.detect() call on a frame already in memory: preprocessing (letterbox, "
                         "colour conversion, tensor), inference and postprocessing (decoding, NMS, scaling boxes), "
                         "plus building Detection records. The same boundary for every backend.",
    "latency_ms.preprocess_ms / inference_ms / postprocess_ms": "Stage times of the same calls as reported by the "
        "backend. ONNX: measured by EdgeVision around preprocessing, session.run() and decoding/NMS. PyTorch: "
        "measured by Ultralytics (Results.speed; synchronises the GPU when one is used). They do not add up exactly "
        "to detect: validation, conversion to Detection records and (PyTorch) Ultralytics' per-call setup are outside.",
    "latency_ms.decode": "Pipeline scope: reading and decoding the next video frame (OpenCV).",
    "latency_ms.track": "Pipeline scope: ByteTrack association for one frame, when tracking is enabled.",
    "latency_ms.frame": "Pipeline scope: decode + detect (+ track) for one frame. Drawing, video writing and JSON "
                        "output are never included.",
    "throughput_fps": "Measured iterations divided by the wall-clock time of the measured loop (frames per second, "
                      "batch size 1, frames processed one after another). Not the same as 1000 / mean latency.",
    "statistics": "count, mean, median (p50), p90, p95, min, max, stdev (sample standard deviation, n - 1; null "
                  "for a single sample). Percentiles use linear interpolation between the nearest ranks.",
}

LIMITATIONS = [
    "Results apply only to the recorded hardware, operating system, library versions, model, input and settings.",
    "Laptop CPUs change speed with power mode, battery vs mains power, temperature and background load; "
    "none of these are controlled or recorded.",
    "Thread counts are the libraries' defaults; PyTorch and ONNX Runtime may use different numbers of threads.",
    "Memory use is not measured.",
    "Python's garbage collector stays enabled, as in normal use.",
    "Detection counts are reported only to show that the workload was the same; they are not an accuracy measure.",
]


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def display_path(path: str | os.PathLike[str]) -> str:
    """``path`` relative to the working directory (POSIX style), or just its file name if outside it."""
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except (ValueError, OSError):
        return candidate.name


def file_identity(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Display path, size and SHA-256 of a file, so runs on the same file can be recognised."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {"path": display_path(path), "size_bytes": os.path.getsize(path), "sha256": digest.hexdigest()}


def git_revision() -> dict[str, Any] | None:
    """Current git commit and whether there are uncommitted changes, if git is available."""
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.SubprocessError):
        return None
    return {"commit": commit, "uncommitted_changes": dirty}


def environment_snapshot(backend: str) -> dict[str, Any]:
    """Hardware and software facts from edgevision.device, without personal paths.

    Only the libraries of the benchmarked backend are queried, so that
    benchmarking ONNX Runtime does not import PyTorch.
    """
    from edgevision import device

    env: dict[str, Any] = {
        "os": device.probe_os(),
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation()},
        "cpu": device.probe_cpu(),
        "nvidia_gpus": [g["name"] for g in device.probe_nvidia_gpus().get("gpus", [])],
        "packages": device.probe_packages(),
    }
    if backend == "pytorch":
        torch_info = device.probe_torch()
        try:
            import torch

            torch_info["intra_op_threads"] = torch.get_num_threads()
        except Exception:  # noqa: BLE001 - optional detail
            pass
        env["torch"] = torch_info
    else:
        env["onnxruntime"] = device.probe_onnxruntime()
        env["onnxruntime"]["intra_op_threads"] = "library default (SessionOptions.intra_op_num_threads = 0)"
    return env


# ---------------------------------------------------------------------------
# Interrupted measurements
# ---------------------------------------------------------------------------

#: An iteration is flagged as suspect when it is both this many times slower than the
#: run's median and longer than SUSPECT_MIN_MS. Ordinary jitter (a few times the median)
#: is never flagged; sleep/hibernation, heavy swapping or a debugger pause is.
SUSPECT_FACTOR = 10.0
SUSPECT_MIN_MS = 1000.0


def find_suspect_iterations(
    samples_ms: list[float], factor: float = SUSPECT_FACTOR, min_ms: float = SUSPECT_MIN_MS
) -> list[dict[str, Any]]:
    """Iterations so much slower than the rest that the measurement was probably interrupted.

    The samples are kept exactly as measured; this only flags them, so that a
    result in which, e.g., a laptop went to sleep is not mistaken for a valid one.
    """
    if not samples_ms:
        return []
    median = summarize(samples_ms)["median"]
    limit = max(factor * median, min_ms)
    return [{"iteration": i, "ms": v, "times_median": v / median if median > 0 else None}
            for i, v in enumerate(samples_ms) if v > limit]


def is_suspect_warning(text: str) -> bool:
    return "took far longer than the rest" in text


def suspect_runs(doc: dict[str, Any]) -> list[int]:
    """Runs of a result that contain flagged (probably interrupted) iterations.

    Uses the stored flags, or recomputes them from the raw samples for files
    written before the flags existed.
    """
    flagged = []
    for r in doc.get("runs", []):
        suspects = r.get("suspect_iterations")
        if suspects is None:
            samples = r.get("samples_ms", {})
            suspects = find_suspect_iterations(samples.get("frame") or samples.get("detect") or [])
        if suspects:
            flagged.append(r["run"])
    return flagged


def _suspect_warning(run: int, suspects: list[dict[str, Any]], median: float) -> str:
    worst = max(s["ms"] for s in suspects)
    return (
        f"run {run}: {len(suspects)} iteration(s) took far longer than the rest (longest {worst / 1000:.1f} s; "
        f"median {median:.1f} ms). The computer was probably asleep, suspended or heavily disturbed. Mean, max, "
        "stdev and throughput that include this run are not representative: rerun the benchmark "
        "(with --overwrite) and keep the computer awake"
    )


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------


def summarize_runs(runs: list[RunResult]) -> dict[str, Any]:
    """Statistics over all runs: pooled samples, plus how much the runs differ from each other."""
    metrics = runs[0].samples_ms.keys()
    pooled = {
        name: summarize([v for r in runs for v in r.samples_ms[name]])
        for name in metrics
        if all(name in r.samples_ms for r in runs)
    }
    medians = [summarize(r.samples_ms["detect"])["median"] for r in runs]
    fps = [r.throughput_fps for r in runs]
    return {
        "runs": len(runs),
        "measured_iterations_total": sum(r.measured_iterations for r in runs),
        "latency_ms_pooled": pooled,
        "detect_median_ms_per_run": medians,
        "throughput_fps_per_run": fps,
        "throughput_fps": {"mean": sum(fps) / len(fps), "min": min(fps), "max": max(fps)},
        "init_s_per_run": [r.init_s for r in runs],
        "first_inference_ms_per_run": [r.first_inference_ms for r in runs],
        "detections_total_per_run": [r.detections_total for r in runs],
    }


def build_report(
    *,
    plan: dict[str, Any],
    backend: dict[str, Any],
    input_info: dict[str, Any],
    runs: list[RunResult],
    library_import: dict[str, Any],
    environment: dict[str, Any],
    git: dict[str, Any] | None,
    warnings: list[str],
) -> dict[str, Any]:
    if not runs:
        raise ValueError("a benchmark report needs at least one run")
    run_dicts = [r.to_dict() for r in runs]
    warnings = list(warnings)
    for run, run_dict in zip(runs, run_dicts):
        metric = "frame" if "frame" in run.samples_ms else "detect"
        suspects = find_suspect_iterations(run.samples_ms[metric])
        run_dict["suspect_iterations"] = suspects
        if suspects:
            warnings.append(_suspect_warning(run.run, suspects, summarize(run.samples_ms[metric])["median"]))
    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "edgevision_version": edgevision.__version__,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": git,
        "benchmark": plan,
        "backend": backend,
        "input": input_info,
        "environment": environment,
        "library_import": library_import,
        "summary": summarize_runs(runs),
        "runs": run_dicts,
        "timing_definitions": TIMING_DEFINITIONS,
        "limitations": LIMITATIONS,
        "warnings": warnings,
    }


def write_report(path: Path, document: dict[str, Any], *, overwrite: bool = False) -> None:
    """Write the document as JSON via ``PATH.partial`` + rename. Refuses to replace a file unless asked."""
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError(f"benchmark results must be written to a .json file, got {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    try:
        # allow_nan=False: NaN/Infinity are not valid JSON, so a broken measurement fails loudly.
        partial.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def load_report(path: Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema") != SCHEMA_NAME:
        raise ValueError(f"{path} is not an EdgeVision benchmark result (schema {document.get('schema')!r})")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path} has schema version {document.get('schema_version')!r}; expected {SCHEMA_VERSION}")
    return document


# ---------------------------------------------------------------------------
# Human-readable output
# ---------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _runtime(doc: dict[str, Any]) -> str:
    b = doc["backend"]
    providers = b.get("execution_providers")
    where = f"providers {', '.join(providers)}" if providers else f"device {b['device']}"
    return f"{b['name']} ({where})"


def format_summary(doc: dict[str, Any]) -> str:
    plan, b, s, inp = doc["benchmark"], doc["backend"], doc["summary"], doc["input"]
    imp = doc["library_import"]
    lines = [
        "Benchmark summary (results apply only to this machine and these settings)",
        f"  Backend            {_runtime(doc)}",
        f"  Model              {b['model']['path']}  (input tensor {b['model_input_shape']}, letterbox {b['letterbox']})",
        f"  Input              {inp['kind']} {inp['file']['path']} ({inp['width']} x {inp['height']} px)",
        f"  Settings           conf {b['settings']['conf_threshold']}, IoU {b['settings']['iou_threshold']}, "
        f"classes {b['settings']['classes'] or 'all'}",
        f"  Scope              {plan['scope']}; {plan['runs']} run(s) x ({plan['warmup_iterations']} warm-up + "
        f"{plan['measured_iterations']} measured)",
        f"  CPU                {doc['environment']['cpu'].get('model') or 'unknown'}",
        "",
        f"  Library import     {_fmt(imp['seconds'], 3)} s" + (" (already imported)" if imp["already_imported"] else ""),
        f"  Initialisation     {', '.join(_fmt(v, 3) for v in s['init_s_per_run'])} s (per run)",
        f"  First inference    {', '.join(_fmt(v, 1) for v in s['first_inference_ms_per_run'])} ms (per run)",
        "",
        f"  Latency, all runs pooled (ms)   {'count':>6} {'mean':>8} {'p50':>8} {'p90':>8} {'p95':>8} "
        f"{'min':>8} {'max':>8} {'stdev':>8}",
    ]
    for name, st in s["latency_ms_pooled"].items():
        lines.append(
            f"    {name:<29}{st['count']:>6} {_fmt(st['mean']):>8} {_fmt(st['median']):>8} {_fmt(st['p90']):>8} "
            f"{_fmt(st['p95']):>8} {_fmt(st['min']):>8} {_fmt(st['max']):>8} {_fmt(st['stdev']):>8}"
        )
    tp = s["throughput_fps"]
    lines += [
        "",
        f"  Throughput         {', '.join(_fmt(v) for v in s['throughput_fps_per_run'])} frames/s per run "
        f"(mean {_fmt(tp['mean'])}, min {_fmt(tp['min'])}, max {_fmt(tp['max'])})",
        f"  Median detect      {', '.join(_fmt(v) for v in s['detect_median_ms_per_run'])} ms per run",
        f"  Detections         {', '.join(str(v) for v in s['detections_total_per_run'])} per run "
        "(workload check, not accuracy)",
    ]
    lines += [f"  Note: {n}" for n in sorted({n for r in doc["runs"] for n in r["notes"]})]
    lines += [f"  {'WARNING' if is_suspect_warning(w) else 'Note'}: {w}" for w in doc["warnings"]]
    return "\n".join(lines)


#: Settings that must match for two results to describe the same workload.
COMPARABLE_FIELDS = [
    ("input file (SHA-256)", lambda d: d["input"]["file"]["sha256"]),
    ("input kind", lambda d: d["input"]["kind"]),
    ("scope", lambda d: d["benchmark"]["scope"]),
    ("tracking", lambda d: d["benchmark"]["tracking"]),
    ("warm-up iterations", lambda d: d["benchmark"]["warmup_iterations"]),
    ("measured iterations", lambda d: d["benchmark"]["measured_iterations"]),
    ("runs", lambda d: d["benchmark"]["runs"]),
    ("preloaded frames", lambda d: d["benchmark"]["preload_frames"]),
    ("model input tensor", lambda d: d["backend"]["model_input_shape"]),
    ("input size", lambda d: d["backend"]["settings"]["input_size"]),
    ("confidence threshold", lambda d: d["backend"]["settings"]["conf_threshold"]),
    ("IoU threshold", lambda d: d["backend"]["settings"]["iou_threshold"]),
    ("class filter", lambda d: d["backend"]["settings"]["classes"]),
    ("device", lambda d: d["backend"]["device"]),
    ("CPU", lambda d: d["environment"]["cpu"].get("model")),
    ("operating system", lambda d: d["environment"]["os"].get("platform")),
    ("EdgeVision version", lambda d: d["edgevision_version"]),
]


def comparability_problems(docs: list[dict[str, Any]]) -> list[str]:
    """Settings that differ between results; each one weakens a direct comparison."""
    problems = []
    for label, get in COMPARABLE_FIELDS:
        values = []
        for d in docs:
            try:
                values.append(get(d))
            except (KeyError, TypeError):
                values.append(None)
        if any(v != values[0] for v in values[1:]):
            problems.append(f"{label} differs: " + " vs ".join(json.dumps(v) for v in values))
    return problems


def format_comparison(docs: list[dict[str, Any]], names: list[str]) -> str:
    """Key results side by side, plus every setting that differs. Does not rank the results."""
    rows: list[tuple[str, list[str]]] = []

    def row(label: str, values: list[str]) -> tuple[str, list[str]]:
        rows.append((label, values))
        return label, values

    def metric(doc: dict[str, Any], name: str, key: str) -> str:
        st = doc["summary"]["latency_ms_pooled"].get(name)
        return "n/a" if st is None else _fmt(st[key])

    row("", names)
    row("backend", [d["backend"]["name"] for d in docs])
    row("device / provider", [", ".join(d["backend"].get("execution_providers") or [d["backend"]["device"]])
                              for d in docs])
    row("model input tensor (H x W)", ["x".join(map(str, d["backend"]["model_input_shape"][2:])) for d in docs])
    def mean_init(doc: dict[str, Any]) -> str:
        values = doc["summary"]["init_s_per_run"]
        return _fmt(sum(values) / len(values), 3)

    row("init (s), mean of runs", [mean_init(d) for d in docs])
    for name in ("decode", "detect", "preprocess_ms", "inference_ms", "postprocess_ms", "track", "frame"):
        if any(name in d["summary"]["latency_ms_pooled"] for d in docs):
            for key in ("median", "p95"):
                row(f"{name.removesuffix('_ms')} {key} (ms)", [metric(d, name, key) for d in docs])
    row("throughput (frames/s), mean", [_fmt(d["summary"]["throughput_fps"]["mean"]) for d in docs])
    row("detections in run 1", [str(d["summary"]["detections_total_per_run"][0]) for d in docs])

    label_width = max(len(label) for label, _ in rows) + 2
    width = max(len(v) for _, values in rows for v in values) + 3
    lines = ["Benchmark comparison (side by side; no result is ranked)"]
    lines += [f"  {label:<{label_width}}" + "".join(f"{v:>{width}}" for v in values) for label, values in rows]
    problems = comparability_problems(docs)
    lines.append("")
    for name, doc in zip(names, docs):
        runs = suspect_runs(doc)
        if runs:
            lines.append(f"WARNING: {name} is not reliable: run(s) {', '.join(map(str, runs))} contain iterations that "
                         "were probably interrupted (computer asleep or disturbed). Rerun it before comparing.")
    if problems:
        lines.append("Not directly comparable - these settings differ:")
        lines += [f"  - {p}" for p in problems]
    else:
        lines.append("Same input, settings, iteration counts, device and machine in all results.")
    lines.append("Differences in speed reflect this machine and these settings only.")
    return "\n".join(lines)
