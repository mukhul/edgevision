# scripts/

Command-line entry points. Run them from the repository root after installing the package.

## Implemented

| Script | Purpose |
|--------|---------|
| `detect_image.py` | Run a YOLO detector on one image, with PyTorch (`--backend pytorch`, the default) or ONNX Runtime (`--backend onnx`, an exported `.onnx` file, default `model.onnx_path`); print detections (class, confidence, `xyxy` box in pixels) and optionally save them as JSON. Defaults come from `configs/default.yaml`. Needs the `vision` extra plus `detector` (PyTorch) or `onnxruntime` (ONNX); with PyTorch, the first run downloads `yolov8n.pt` to `models/`. ONNX models are never downloaded. |
| `detect_video.py` | Run a YOLO detector (`--backend pytorch` or `onnx`, as above; `--pytorch-letterbox square` gives PyTorch the same square input as ONNX) on every frame of a video via `edgevision.pipeline`; print a processing summary (frames processed, frames with/without detections, totals per class; no timing) optionally stream per-frame detections to JSON (schema in `results/README.md`), optionally write an annotated `.mp4` (`--output-video`, with `--fps` to override the frame rate), and optionally track objects across frames with ByteTrack (`--tracking`, `--tracker-config`; needs the `tracking` extra), adding track IDs to the video labels and JSON. Video path, frame limit, model and thresholds default to the config. Needs the `vision` extra plus `detector` or `onnxruntime`. |
| `export_onnx.py` | Export a local YOLO `.pt` model to ONNX on the CPU (`--model`, `--output`, `--imgsz`, `--opset`, `--dynamic`, `--no-simplify`, `--overwrite`; defaults from the config), then validate the file structurally (`--no-validate` to skip). Never downloads weights and never replaces an existing file without `--overwrite`. Needs the `detector`, `export` and `onnxruntime` extras (`onnxruntime` is not needed with `--no-simplify`). |
| `inspect_onnx.py` | Validate an existing `.onnx` file with the official ONNX checker and report its opset, inputs, outputs and metadata; `--imgsz N` also checks the input size. Structural only: it does not compare results with PyTorch. Needs the `export` extra (only `onnx` is used). |
| `benchmark.py` | Time one backend (`--backend pytorch\|onnx`, `--device`) on an image or video: initialisation, first inference, per-frame latency statistics (detect and its preprocess / inference / postprocess stages, or decode / detect / track with `--scope pipeline`) and throughput, over `--runs` runs of `--warmup` + `--iterations` iterations. Saves JSON (default `results/benchmarks/...`, never overwritten without `--overwrite`) and prints a summary. Defaults from the config's `benchmark` and `model` sections. Never downloads models. See `docs/benchmarking.md`. |
| `compare_benchmarks.py` | Show two or more benchmark JSON files side by side and list every setting that differs. Does not rank the results. |
| `plot_benchmarks.py` | Validate 1-4 benchmark JSON files and write `latency.png`, `throughput.png`, `stages.png`, `latency_runs.png` and `summary.md` (default `results/plots/`, `--output`, `--overwrite`). Needs the `plots` extra (matplotlib). Performance only. |
| `compare_detections.py` | Agreement between two `detect_video.py --json` files of the same video: matched pairs, detections only one run made, IoU and confidence differences, per class (`--iou`, `--output`). **Not** an accuracy measure. Needs NumPy (installed with the `vision` extra). |
| `evaluate.py` | Accuracy against labelled images: `--annotations` (COCO JSON) and `--images` (folder), any backend; precision, recall, F1, AP / mAP@0.5 / mAP@0.5:0.95 per class and overall, saved to `results/evaluation/`. Needs labelled data you supply; see `docs/evaluation.md`. |
| `inspect_video.py` | Report a video's metadata (resolution, FPS, reported frame count, codec), how many frames can actually be decoded, and whether reading stopped at the end of the video or at `--max-frames`. Needs the `vision` extra. |

```bash
python scripts/detect_image.py path/to/photo.jpg
python scripts/detect_image.py path/to/photo.jpg --device cpu --conf 0.4 --json results/photo.json
python scripts/detect_video.py data/input/sample.mp4
python scripts/detect_video.py data/input/sample.mp4 --max-frames 100 --json results/sample_detections.json
python scripts/detect_video.py data/input/sample.mp4 --all-frames --device cpu
python scripts/detect_video.py data/input/sample.mp4 --output-video results/sample_annotated.mp4
python scripts/detect_video.py data/input/sample.mp4 --tracking --output-video results/sample_tracked.mp4 --json results/sample_tracks.json
python scripts/detect_image.py path/to/photo.jpg --backend onnx --device cpu
python scripts/detect_video.py data/input/sample.mp4 --backend onnx --json results/sample_onnx.json
python scripts/inspect_video.py data/input/sample.mp4
python scripts/inspect_video.py data/input/sample.mp4 --max-frames 100
python scripts/inspect_video.py --help
python scripts/export_onnx.py
python scripts/export_onnx.py --imgsz 320 --output models/yolov8n_320.onnx
python scripts/inspect_onnx.py models/yolov8n.onnx --imgsz 640
python scripts/benchmark.py --backend pytorch --device cpu --output results/benchmarks/pytorch_cpu.json
python scripts/benchmark.py --backend onnx --device cpu --output results/benchmarks/onnx_cpu.json
python scripts/compare_benchmarks.py results/benchmarks/pytorch_cpu.json results/benchmarks/onnx_cpu.json
python scripts/plot_benchmarks.py results/benchmarks/pytorch_cpu.json results/benchmarks/onnx_cpu.json
python scripts/compare_detections.py results/sample_detections.json results/sample_onnx.json
python scripts/evaluate.py --annotations data/eval/annotations.json --images data/eval/images
```

`plot_benchmarks.py`, `compare_detections.py`, `evaluate.py`: `0` success, `1` unusable input file, missing package/model/images, or output exists, `2` invalid arguments (`evaluate.py`: `130` interrupted).

`benchmark.py`: `0` success, `1` failure (missing model, input or package, error during a run, output exists), `2` invalid arguments, `130` interrupted (nothing written). `compare_benchmarks.py`: `0`, `1` unreadable file, `2` fewer than two files.

`export_onnx.py` and `inspect_onnx.py`: `0` success, `1` failure (missing model or package, output already exists, export failed, invalid ONNX file, unexpected layout), `2` invalid arguments. `--help` never exports.

Exit status for the other three: `0` success, `1` the input cannot be read or processed (missing, corrupt, missing package, model or inference failure, CUDA requested but unavailable, output file already exists, odd-sized video for `--output-video`), `2` invalid arguments (including `--output-video` equal to the input or to `--json`). `detect_video.py` returns `130` when interrupted with Ctrl+C.

All scripts are intended to work in both POSIX shells and Windows PowerShell / Command Prompt.
