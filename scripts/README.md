# scripts/

Command-line entry points. Run them from the repository root after installing the package.

## Implemented

| Script | Purpose |
|--------|---------|
| `inspect_video.py` | Report a video's metadata (resolution, FPS, reported frame count, codec), how many frames can actually be decoded, and whether reading stopped at the end of the video or at `--max-frames`. Needs the `vision` extra. |

```bash
python scripts/inspect_video.py data/input/sample.mp4
python scripts/inspect_video.py data/input/sample.mp4 --max-frames 100
python scripts/inspect_video.py --help
```

Exit status: `0` success, `1` the video cannot be read (missing, not a file, corrupt, OpenCV not installed), `2` invalid arguments.

## Planned

| Script | Purpose |
|--------|---------|
| `export_onnx.py` | Export the PyTorch model to ONNX and validate it against PyTorch outputs |
| `run_video.py` | Process a video with a chosen backend and write an annotated output video |
| `benchmark.py` | Run repeatable latency/throughput benchmarks across backends |
| `evaluate.py` | Evaluate detection quality against ground-truth annotations |
| `plot_results.py` | Generate plots from saved benchmark results |

All scripts are intended to work in both POSIX shells and Windows PowerShell / Command Prompt.
