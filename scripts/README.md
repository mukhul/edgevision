# scripts/

Command-line entry points. **None of these exist yet**; this directory is reserved for them.

| Planned script | Purpose |
|----------------|---------|
| `export_onnx.py` | Export the PyTorch model to ONNX and validate it against PyTorch outputs |
| `run_video.py` | Process a video with a chosen backend and write an annotated output video |
| `benchmark.py` | Run repeatable latency/throughput benchmarks across backends |
| `evaluate.py` | Evaluate detection quality against ground-truth annotations |
| `plot_results.py` | Generate plots from saved benchmark results |

All scripts are intended to run from the repository root and to work in both POSIX shells and Windows PowerShell.
