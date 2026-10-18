"""Plot benchmark result files and write a summary table.

Run from the repository root:

    python scripts/plot_benchmarks.py results/benchmarks/pytorch_cpu.json results/benchmarks/onnx_cpu.json
    python scripts/plot_benchmarks.py results/benchmarks/*.json --output results/plots/laptop_cpu

Writes latency.png, throughput.png, stages.png, latency_runs.png and
summary.md to --output (default results/plots). Existing files are only
replaced with --overwrite. Every figure states the machine and workload and
prints any setting that differs between the files. These are performance
plots; they say nothing about detection accuracy.

Exit status: 0 success, 1 unusable input file or output exists, 2 invalid arguments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from edgevision.evaluation.benchmark_plots import BenchmarkFileError, generate_report, load_benchmark_files
from edgevision.optional import MissingDependencyError

DEFAULT_OUTPUT = Path("results") / "plots"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/plot_benchmarks.py",
        description="Plot latency, throughput and stage breakdown from benchmark JSON files.",
    )
    parser.add_argument("results", type=Path, nargs="+", help="1-4 JSON files from scripts/benchmark.py")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"folder for the figures and summary.md (default: {DEFAULT_OUTPUT.as_posix()})")
    parser.add_argument("--overwrite", action="store_true", help="replace figures already in the output folder")
    args = parser.parse_args(argv)
    try:
        results = load_benchmark_files(args.results)
        written = generate_report(results, args.output, overwrite=args.overwrite)
    except (BenchmarkFileError, FileExistsError, MissingDependencyError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("Wrote:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
