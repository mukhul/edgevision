# scripts/

Command-line entry points. Run them from the repository root after installing the package.

## Implemented

| Script | Purpose |
|--------|---------|
| `detect_image.py` | Run the PyTorch YOLO detector on one image; print detections (class, confidence, `xyxy` box in pixels) and optionally save them as JSON. Defaults come from `configs/default.yaml`. Needs the `vision` and `detector` extras; the first run downloads `yolov8n.pt` to `models/`. |
| `inspect_video.py` | Report a video's metadata (resolution, FPS, reported frame count, codec), how many frames can actually be decoded, and whether reading stopped at the end of the video or at `--max-frames`. Needs the `vision` extra. |

```bash
python scripts/detect_image.py path/to/photo.jpg
python scripts/detect_image.py path/to/photo.jpg --device cpu --conf 0.4 --json results/photo.json
python scripts/inspect_video.py data/input/sample.mp4
python scripts/inspect_video.py data/input/sample.mp4 --max-frames 100
python scripts/inspect_video.py --help
```

Exit status for both: `0` success, `1` the input cannot be read or processed (missing, corrupt, missing package, model or inference failure, CUDA requested but unavailable), `2` invalid arguments.

## Planned

| Script | Purpose |
|--------|---------|
| `export_onnx.py` | Export the PyTorch model to ONNX and validate it against PyTorch outputs |
| `run_video.py` | Process a video with a chosen backend and write an annotated output video |
| `benchmark.py` | Run repeatable latency/throughput benchmarks across backends |
| `evaluate.py` | Evaluate detection quality against ground-truth annotations |
| `plot_results.py` | Generate plots from saved benchmark results |

All scripts are intended to work in both POSIX shells and Windows PowerShell / Command Prompt.
