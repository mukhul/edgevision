"""Show two or more benchmark result files side by side.

Run from the repository root:

    python scripts/compare_benchmarks.py results/benchmarks/pytorch.json results/benchmarks/onnx.json

Prints the main measurements next to each other and lists every setting that
differs between the files (input, iteration counts, thresholds, input tensor,
device, machine...), because results measured under different conditions are
not directly comparable. It does not rank the results or pick a winner.

Exit status: 0 success, 1 a file is missing or not a benchmark result, 2 invalid arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from edgevision.benchmarks.report import format_comparison, load_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/compare_benchmarks.py",
        description="Show benchmark results side by side and list the settings that differ.",
    )
    parser.add_argument("results", type=Path, nargs="+", help="two or more JSON files from scripts/benchmark.py")
    args = parser.parse_args(argv)
    if len(args.results) < 2:
        parser.error("give at least two result files to compare")
    docs = []
    for path in args.results:
        try:
            docs.append(load_report(path))
        except FileNotFoundError:
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            return 1
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"ERROR: could not read {path}: {exc}", file=sys.stderr)
            return 1
    try:
        print(format_comparison(docs, [p.stem for p in args.results]))
    except (KeyError, TypeError, IndexError) as exc:
        print(f"ERROR: a result file is incomplete ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
