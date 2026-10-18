"""Validate benchmark result files and turn them into plots and a summary table.

Inputs are JSON files from ``scripts/benchmark.py`` (schema
``edgevision.benchmark`` v1). Outputs, all in one folder:

* ``latency.png``      median and p95 ``detect`` latency per file, with each run's median as a dot
* ``throughput.png``   mean throughput per file, with each run as a dot
* ``stages.png``       mean preprocess / inference / postprocess / other time per frame
* ``latency_runs.png`` the full distribution of ``detect`` latency, one box per run
* ``summary.md``       every plotted value in a table, plus comparability notes

Rules: a missing metric is drawn and written as "not reported", never as 0;
no sample is removed (iterations flagged as interrupted are kept and
called out on the figure); every figure states the machine and workload; and
any setting that differs between files is printed on every figure. These
are performance measurements of one machine. Nothing here measures accuracy.

Needs matplotlib (the ``plots`` extra). It uses the non-interactive "Agg"
backend, so no window is opened and it works without a display.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from edgevision.benchmarks.report import comparability_problems, find_suspect_iterations, load_report, suspect_runs
from edgevision.optional import import_optional

#: Fixed categorical order (validated palette: CVD-safe, >= 3:1 on the light surface).
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6")
STAGE_COLORS = {"preprocess_ms": "#1baf7a", "inference_ms": "#6250d6", "postprocess_ms": "#eda100",
                "other": "#b4b2ab"}
STAGE_LABELS = {"preprocess_ms": "preprocess", "inference_ms": "inference", "postprocess_ms": "postprocess",
                "other": "other (outside the stages)"}
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
OUTPUT_FILES = ("latency.png", "throughput.png", "stages.png", "latency_runs.png", "summary.md")
MAX_FILES = len(SERIES_COLORS)


class BenchmarkFileError(ValueError):
    """A benchmark result file cannot be used for analysis."""


@dataclass(frozen=True)
class BenchmarkResult:
    name: str  # short label, from the file name
    path: Path
    doc: dict[str, Any]

    @property
    def runtime(self) -> str:
        b = self.doc["backend"]
        providers = b.get("execution_providers")
        return f"{b['name']} · {', '.join(providers) if providers else b['device']}"


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = (
    ("backend", "name"), ("backend", "device"), ("backend", "model_input_shape"), ("backend", "settings"),
    ("benchmark", "scope"), ("benchmark", "warmup_iterations"), ("benchmark", "measured_iterations"),
    ("benchmark", "runs"), ("input", "kind"), ("input", "file", "path"), ("input", "width"), ("input", "height"),
    ("environment", "cpu"), ("summary", "latency_ms_pooled", "detect"), ("summary", "throughput_fps_per_run"),
    ("summary", "init_s_per_run"), ("summary", "detections_total_per_run"), ("runs",),
)
STAT_KEYS = ("count", "mean", "median", "p90", "p95", "min", "max")


def _get(doc: Any, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if not isinstance(doc, dict) or key not in doc:
            raise KeyError(".".join(keys))
        doc = doc[key]
    return doc


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_benchmark(doc: dict[str, Any]) -> list[str]:
    """Every problem that prevents analysing ``doc`` (empty if it is usable)."""
    problems = []
    for keys in REQUIRED_FIELDS:
        try:
            _get(doc, keys)
        except KeyError:
            problems.append(f"missing field {'.'.join(keys)}")
    if problems:
        return problems
    detect = doc["summary"]["latency_ms_pooled"]["detect"]
    for key in STAT_KEYS:
        if not _finite(detect.get(key)):
            problems.append(f"summary.latency_ms_pooled.detect.{key} is missing or not a finite number")
    runs = doc["runs"]
    if not isinstance(runs, list) or not runs:
        problems.append("runs must be a non-empty list")
        return problems
    for i, run in enumerate(runs):
        samples = run.get("samples_ms", {}).get("detect") if isinstance(run, dict) else None
        if not isinstance(samples, list) or not samples or not all(_finite(v) for v in samples):
            problems.append(f"runs[{i}].samples_ms.detect must be a non-empty list of finite numbers")
        if not isinstance(run, dict) or not _finite(run.get("throughput_fps")):
            problems.append(f"runs[{i}].throughput_fps is missing or not a finite number")
    fps = doc["summary"]["throughput_fps_per_run"]
    if not isinstance(fps, list) or len(fps) != len(runs) or not all(_finite(v) for v in fps):
        problems.append("summary.throughput_fps_per_run must hold one finite number per run")
    return problems


def load_benchmark_files(paths: list[str | os.PathLike[str]]) -> list[BenchmarkResult]:
    """Load and validate result files; raise BenchmarkFileError naming the file and every problem."""
    if not paths:
        raise BenchmarkFileError("give at least one benchmark result file")
    if len(paths) > MAX_FILES:
        raise BenchmarkFileError(f"at most {MAX_FILES} result files can be plotted together, got {len(paths)}")
    results = []
    for p in paths:
        path = Path(p)
        if not path.is_file():
            raise BenchmarkFileError(f"Benchmark file not found: {path}")
        try:
            doc = load_report(path)
        except ValueError as exc:  # includes json.JSONDecodeError
            raise BenchmarkFileError(f"{path}: {exc}") from exc
        problems = validate_benchmark(doc)
        if problems:
            raise BenchmarkFileError(f"{path} cannot be analysed:\n" + "\n".join(f"  - {x}" for x in problems))
        results.append(BenchmarkResult(path.stem, path, doc))
    names = [r.name for r in results]
    if len(set(names)) != len(names):  # same file name in different folders: use the folder too
        results = [BenchmarkResult(f"{r.path.parent.name}/{r.name}", r.path, r.doc) for r in results]
    return results


# ---------------------------------------------------------------------------
# Values used by the plots and the table (None = not reported)
# ---------------------------------------------------------------------------


def stage_means(doc: dict[str, Any]) -> dict[str, float] | None:
    """Mean ms per frame of each stage plus 'other' (detect minus the stages), or None if not reported."""
    pooled = doc["summary"]["latency_ms_pooled"]
    if not all(k in pooled for k in ("preprocess_ms", "inference_ms", "postprocess_ms")):
        return None
    means = {k: float(pooled[k]["mean"]) for k in ("preprocess_ms", "inference_ms", "postprocess_ms")}
    means["other"] = float(pooled["detect"]["mean"]) - sum(means.values())
    return means


def run_medians(doc: dict[str, Any]) -> list[float]:
    from edgevision.benchmarks.stats import summarize

    return [summarize(r["samples_ms"]["detect"])["median"] for r in doc["runs"]]


def recorded_hours_apart(results: list[BenchmarkResult]) -> float | None:
    """Hours between the earliest and latest ``created_utc``; None if unknown or a single file."""
    from datetime import datetime

    stamps = []
    for r in results:
        try:
            stamps.append(datetime.strptime(r.doc["created_utc"], "%Y-%m-%dT%H:%M:%SZ"))
        except (KeyError, TypeError, ValueError):
            return None
    return (max(stamps) - min(stamps)).total_seconds() / 3600 if len(stamps) > 1 else None


def context_lines(results: list[BenchmarkResult]) -> tuple[str, list[str]]:
    """A subtitle describing the shared workload, and notes about anything that differs or is suspect."""
    d = results[0].doc
    b, plan, inp = d["backend"], d["benchmark"], d["input"]
    subtitle = (
        f"{d['environment']['cpu'].get('model') or 'unknown CPU'} · {plan['scope']} scope · "
        f"{inp['file']['path']} ({inp['width']}×{inp['height']}) · input tensor {'×'.join(map(str, b['model_input_shape']))} · "
        f"{plan['runs']} runs × {plan['measured_iterations']} measured (+{plan['warmup_iterations']} warm-up)"
    )
    notes = []
    problems = comparability_problems([r.doc for r in results]) if len(results) > 1 else []
    if problems:
        subtitle = f"{results[0].name}: " + subtitle
        notes.append("NOT DIRECTLY COMPARABLE - " + "; ".join(problems))
    hours = recorded_hours_apart(results)
    if hours is not None and hours >= 1:
        notes.append(f"The results were recorded {hours:.1f} h apart: temperature, power state and background load "
                     "may have differed. For the fairest comparison, run the backends back to back.")
    for r in results:
        flagged = suspect_runs(r.doc)
        if flagged:
            notes.append(f"{r.name}: run(s) {', '.join(map(str, flagged))} contain iterations flagged as interrupted "
                         "(computer asleep or disturbed); kept in the data - rerun before drawing conclusions")
    notes.append("Measured on one machine with these settings; not a general result. "
                 "Performance only - nothing here measures detection accuracy.")
    return subtitle, notes


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _pyplot() -> Any:
    matplotlib = import_optional("matplotlib", display_name="matplotlib", extra="plots", purpose="draw plots")
    matplotlib.use("Agg")  # no GUI, works without a display (CI, remote shells)
    import matplotlib.pyplot as plt

    return plt


def _figure(plt: Any, title: str, subtitle: str, notes: list[str], size: tuple[float, float] = (9.5, 5.6),
            left: float = 0.08) -> tuple:
    import textwrap

    fig, ax = plt.subplots(figsize=size, dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    wrapped_notes = [textwrap.fill(n, 150) for n in notes]
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.915, textwrap.fill(subtitle, 140), ha="left", va="top", fontsize=8, color=MUTED)
    fig.text(0.01, 0.01, "\n".join(wrapped_notes), ha="left", va="bottom", fontsize=7, color=MUTED)
    bottom = 0.09 + 0.022 * sum(n.count("\n") + 1 for n in wrapped_notes)
    fig.subplots_adjust(left=left, right=0.98, top=0.82, bottom=bottom + 0.05)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig: Any, plt: Any, path: Path) -> None:
    partial = path.with_name(path.stem + ".partial" + path.suffix)
    try:
        fig.savefig(partial, facecolor=fig.get_facecolor())
        os.replace(partial, path)
    finally:
        plt.close(fig)
        partial.unlink(missing_ok=True)


def _tick_labels(results: list[BenchmarkResult]) -> list[str]:
    return [f"{r.name}\n{r.runtime}" for r in results]


def plot_latency(results: list[BenchmarkResult], path: Path) -> None:
    plt = _pyplot()
    subtitle, notes = context_lines(results)
    fig, ax = _figure(plt, "Per-frame detect() latency: median and p95 (all runs pooled)", subtitle, notes)
    width = 0.34
    top = 0.0
    for i, r in enumerate(results):
        st = r.doc["summary"]["latency_ms_pooled"]["detect"]
        color = SERIES_COLORS[i]
        medians = run_medians(r.doc)
        for offset, key, kwargs in ((-width / 2, "median", {}), (width / 2, "p95", {"hatch": "////", "alpha": 0.55})):
            value = float(st[key])
            ax.bar(i + offset, value, width * 0.94, color=color, edgecolor=SURFACE, linewidth=1.5, **kwargs)
            label_at = max([value] + (medians if key == "median" else []))  # above the run dots
            ax.annotate(f"{key}\n{value:.1f}", (i + offset, label_at), xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=INK)
            top = max(top, label_at)
        ax.scatter([i - width / 2] * len(medians), medians, s=28, zorder=3, facecolor=SURFACE, edgecolor=INK,
                   linewidth=1, label="median of each run" if i == 0 else None)
    ax.set_xticks(range(len(results)), _tick_labels(results), fontsize=8, color=INK)
    ax.set_ylabel("milliseconds per frame (lower = less time)", fontsize=9, color=MUTED)
    ax.set_ylim(0, top * 1.22)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    _save(fig, plt, path)


def plot_throughput(results: list[BenchmarkResult], path: Path) -> None:
    plt = _pyplot()
    subtitle, notes = context_lines(results)
    fig, ax = _figure(plt, "Throughput: frames per second over the measured loop", subtitle, notes)
    top = 0.0
    for i, r in enumerate(results):
        fps = [float(v) for v in r.doc["summary"]["throughput_fps_per_run"]]
        mean = sum(fps) / len(fps)
        ax.bar(i, mean, 0.5, color=SERIES_COLORS[i], edgecolor=SURFACE, linewidth=1.5)
        ax.scatter([i] * len(fps), fps, s=28, zorder=3, facecolor=SURFACE, edgecolor=INK, linewidth=1,
                   label="each run" if i == 0 else None)
        ax.annotate(f"mean {mean:.2f}\nruns {min(fps):.2f}–{max(fps):.2f}", (i, max(fps)), xytext=(0, 7),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK)
        top = max(top, max(fps))
    ax.set_xticks(range(len(results)), _tick_labels(results), fontsize=8, color=INK)
    ax.set_ylabel("frames per second (batch 1, one after another)", fontsize=9, color=MUTED)
    ax.set_ylim(0, top * 1.25)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    _save(fig, plt, path)


def plot_stages(results: list[BenchmarkResult], path: Path) -> None:
    plt = _pyplot()
    subtitle, notes = context_lines(results)
    notes = ["Means are shown because they add up (medians do not). Stage boundaries: ONNX Runtime stages are "
             "timed by EdgeVision, PyTorch stages by Ultralytics; 'other' is the rest of detect() (input checks, "
             "building Detection records, per-call setup)."] + notes
    fig, ax = _figure(plt, "Where a frame's time goes: mean ms per stage", subtitle, notes, left=0.22)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    widest = 0.0
    for i, r in enumerate(results):
        means = stage_means(r.doc)
        detect_mean = float(r.doc["summary"]["latency_ms_pooled"]["detect"]["mean"])
        widest = max(widest, detect_mean)
        if means is None:
            ax.barh(i, detect_mean, 0.5, color="#d9d8d3", hatch="////", edgecolor=SURFACE)
            ax.text(detect_mean, i, f"  detect {detect_mean:.1f} ms - stage breakdown not reported", va="center",
                    fontsize=8, color=INK)
            continue
        left = 0.0
        for stage in ("preprocess_ms", "inference_ms", "postprocess_ms", "other"):
            value = means[stage]
            if value <= 0:
                if value < 0:
                    ax.text(left, i + 0.32, f"stages exceed the detect mean by {-value:.2f} ms", fontsize=7,
                            color=MUTED)
                continue
            ax.barh(i, value, 0.5, left=left, color=STAGE_COLORS[stage], edgecolor=SURFACE, linewidth=2,
                    label=STAGE_LABELS[stage] if stage not in ax.get_legend_handles_labels()[1] else None)
            if value >= 0.06 * detect_mean:
                ax.text(left + value / 2, i, f"{value:.1f}", ha="center", va="center", fontsize=8, color=INK)
            left += value
        ax.text(left, i, f"  total {detect_mean:.1f} ms", va="center", fontsize=8, color=INK)
        parts = " · ".join(f"{STAGE_LABELS[k].split(' ')[0]} {means[k]:.1f}" for k in STAGE_LABELS)
        ax.text(0, i + 0.4, f"{parts} ms", va="center", fontsize=7.5, color=MUTED)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, frameon=False,
              ncols=4)
    ax.set_yticks(range(len(results)), _tick_labels(results), fontsize=8, color=INK)
    ax.set_ylim(len(results) - 0.35, -0.45)  # inverted: first file on top, room for the breakdown text
    ax.set_xlim(0, widest * 1.3)
    ax.set_xlabel("milliseconds per frame (mean)", fontsize=9, color=MUTED)
    _save(fig, plt, path)


def plot_run_distributions(results: list[BenchmarkResult], path: Path) -> None:
    plt = _pyplot()
    subtitle, notes = context_lines(results)
    notes = ["Box = middle 50% of iterations, line = median, whiskers = 1.5 × box height, points = every iteration "
             "outside the whiskers (none are removed)."] + notes
    fig, ax = _figure(plt, "detect() latency per run: full distribution", subtitle, notes)
    data, positions, colors, labels = [], [], [], []
    normal_max, flagged = 0.0, 0
    pos = 0.0
    for i, r in enumerate(results):
        for run in r.doc["runs"]:
            samples = [float(v) for v in run["samples_ms"]["detect"]]
            suspects = {s["iteration"] for s in find_suspect_iterations(samples)}
            flagged += len(suspects)
            normal_max = max([normal_max] + [v for j, v in enumerate(samples) if j not in suspects])
            data.append(samples)
            positions.append(pos)
            colors.append(SERIES_COLORS[i])
            labels.append(f"{r.name}\nrun {run['run']}")
            pos += 1
        pos += 0.6
    boxes = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True, showfliers=True,
                       medianprops={"color": INK, "linewidth": 1.5},
                       flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": MUTED, "markeredgewidth": 0})
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor(SURFACE)
    ax.set_xticks(positions, labels, fontsize=7, color=INK)
    ax.set_ylabel("milliseconds per frame", fontsize=9, color=MUTED)
    ax.set_ylim(0, normal_max * 1.15)
    if flagged:
        ax.text(0.5, 0.97, f"{flagged} iteration(s) flagged as interrupted lie above this axis (kept in the data)",
                transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#d03b3b")
    _save(fig, plt, path)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 2) -> str:
    return "not reported" if value is None else f"{value:.{digits}f}"


def summary_markdown(results: list[BenchmarkResult]) -> str:
    subtitle, notes = context_lines(results)
    names = [r.name for r in results]
    rows: list[tuple[str, list[str]]] = []

    def stat(r: BenchmarkResult, metric: str, key: str) -> str:
        st = r.doc["summary"]["latency_ms_pooled"].get(metric)
        return _fmt(None if st is None else st.get(key))

    rows.append(("Backend / provider", [r.runtime for r in results]))
    rows.append(("Model", [r.doc["backend"].get("model", {}).get("path", "not reported") for r in results]))
    rows.append(("Input tensor", ["×".join(map(str, r.doc["backend"]["model_input_shape"])) for r in results]))
    rows.append(("Initialisation, mean of runs (s)",
                 [_fmt(sum(v := r.doc["summary"]["init_s_per_run"]) / len(v), 3) for r in results]))
    rows.append(("First inference, run 1 (ms)",
                 [_fmt((r.doc["summary"].get("first_inference_ms_per_run") or [None])[0], 1) for r in results]))
    for key, label in (("median", "median"), ("p95", "p95"), ("mean", "mean"), ("stdev", "stdev")):
        rows.append((f"detect {label} (ms)", [stat(r, "detect", key) for r in results]))
    for stage in ("preprocess_ms", "inference_ms", "postprocess_ms"):
        rows.append((f"{stage.removesuffix('_ms')} median (ms)", [stat(r, stage, "median") for r in results]))
    rows.append(("detect median per run (ms)", [", ".join(f"{m:.2f}" for m in run_medians(r.doc)) for r in results]))
    rows.append(("Throughput per run (frames/s)",
                 [", ".join(f"{v:.2f}" for v in r.doc["summary"]["throughput_fps_per_run"]) for r in results]))
    rows.append(("Detections per run (workload check)",
                 [", ".join(map(str, r.doc["summary"]["detections_total_per_run"])) for r in results]))
    rows.append(("Runs flagged as interrupted", [", ".join(map(str, suspect_runs(r.doc))) or "none" for r in results]))

    lines = ["# Benchmark summary", "", subtitle, "",
             "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    lines += [f"| {label} | " + " | ".join(values) + " |" for label, values in rows]
    lines += ["", "## Notes", ""] + [f"- {n}" for n in notes] + [
        "- Detection counts only confirm that the same work was done; they are not an accuracy measure.",
        "- Timing definitions: see docs/benchmarking.md. Source files:",
    ] + [f"  - `{r.name}`: {r.path.name}, created {r.doc.get('created_utc', 'unknown')}" for r in results]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Everything at once
# ---------------------------------------------------------------------------


def generate_report(results: list[BenchmarkResult], output_dir: Path, *, overwrite: bool = False) -> list[Path]:
    """Write all figures and summary.md to ``output_dir``. Refuses to replace files unless ``overwrite``."""
    output_dir = Path(output_dir)
    existing = [output_dir / f for f in OUTPUT_FILES if (output_dir / f).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{output_dir} already contains {', '.join(p.name for p in existing)}; "
                              "pass --overwrite or choose another --output folder")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_latency(results, output_dir / "latency.png")
    plot_throughput(results, output_dir / "throughput.png")
    plot_stages(results, output_dir / "stages.png")
    plot_run_distributions(results, output_dir / "latency_runs.png")
    summary = output_dir / "summary.md"
    summary.write_text(summary_markdown(results), encoding="utf-8")
    return [output_dir / f for f in OUTPUT_FILES]
