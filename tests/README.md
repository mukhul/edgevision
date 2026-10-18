# tests/

Automated tests (pytest). Run them from the repository root after `python -m pip install -e ".[dev]"`:

```bash
python -m pytest            # whole suite
python -m pytest -v         # one line per test
python -m pytest tests/test_config.py
```

The suite needs no GPU, no network access, no downloaded models, and none of the optional ML packages. PyTorch, ONNX, ONNX Runtime, OpenCV, Ultralytics and `nvidia-smi` are replaced with small fakes where a test needs them (the shared fake YOLO model is in `conftest.py`).

A few **integration tests** use real OpenCV to write and read back a tiny synthetic video. Install the `vision` extra (`python -m pip install -e ".[dev,vision]"`) to run them; without it they are reported as *skipped* with the reason, and every other test still runs. Likewise, ONNX validation tests on small real ONNX files need the `export` extra and are skipped without it (a fake `onnx` module covers the same logic either way).

**Mocked tests are not real inference.** They check EdgeVision's own code around the model. Six opt-in tests (marker `model`) use the genuine pretrained YOLOv8n on CPU: on `bus.jpg`, which ships inside the Ultralytics package, on a short video built from it, on that video with real ByteTrack (the bus must keep one track ID), a real ONNX export at 320×320 into a temporary folder followed by validation, `models/yolov8n.onnx` on ONNX Runtime compared with Ultralytics running the same file (same detections, boxes within 0.01 px), and a tiny real benchmark of both backends (2 warm-up + 3 measured iterations; checks the result files and that both backends found the same detections, never the speed). They need the `detector` extra (plus `tracking` / `export` / `onnxruntime` for those tests) and an existing `models/yolov8n.pt` (they never download):

```bash
EDGEVISION_RUN_MODEL_TESTS=1 python -m pytest -m model -v      # Linux / macOS
```

On Windows Command Prompt: `set EDGEVISION_RUN_MODEL_TESTS=1`, then `python -m pytest -m model -v`. `EDGEVISION_MODEL_WEIGHTS` can point to weights elsewhere.

## Current coverage

| File | What it checks |
|------|----------------|
| `test_package.py` | The package is installed and imported from `src/` |
| `test_config.py` | Valid configs load; missing files, malformed YAML, duplicate keys, missing/unknown fields, wrong types and out-of-range values are rejected with clear messages |
| `test_device.py` | The environment report runs without optional packages; CUDA/provider detection, broken installs, `nvidia-smi` parsing and failures, and setup warnings |
| `test_video_reader.py` | Missing/directory/unopenable/undecodable inputs, metadata (including invalid values), frame order and indices, `max_frames`, end of video, capture release on every exit path, missing or broken OpenCV; plus integration tests on a synthetic video |
| `test_inspect_video.py` | `scripts/inspect_video.py` as a real command: usage, exit codes, error messages, metadata and stop-reason output |
| `test_detection.py` | `Detection` validation (coordinates, confidence, IDs, names, booleans), `DetectionSettings` validation and sync with the default config, frame validation |
| `test_pytorch_backend.py` | Arguments passed to Ultralytics, output conversion, frame-index propagation, empty results, invalid inputs, device selection (including CUDA requested but unavailable), load/inference failures, missing packages, closing; plus the opt-in real-model test |
| `test_detect_image.py` | `scripts/detect_image.py`: arguments, config defaults and overrides, output, JSON export and overwrite protection, failure exit codes |
| `test_pipeline.py` | Frame order and indices reaching the detector, tracker integration (IDs attached in order, reset per video, failures reported, tracker may only add IDs, detection-only output unchanged), aggregation and summary consistency, frames/videos without detections, empty video, `max_frames`, detector failures with context, contract checks, video released on success/failure/early stop; integration with the real `VideoReader`; opt-in real-model video test |
| `test_results.py` | JSON schema, parseable output, `.partial`-then-rename behaviour, no file after errors, overwrite protection, invalid values rejected |
| `test_detect_video.py` | `scripts/detect_video.py`: arguments, config defaults vs `--max-frames`/`--all-frames`, summary, JSON contents, overwrite protection, inference failure and Ctrl+C leave no results file; annotated video size, frame count and FPS (source and `--fps`), JSON unchanged by `--output-video`, path conflicts, failures mid-video leave no files, odd sizes rejected; `--tracking` / `--no-tracking` / `--tracker-config`, track IDs in JSON only when tracking, tracker failures leave no files |
| `test_annotator.py` | Drawing on real pixels: no detections, one and many detections, labels (via a spy on `cv2.putText`), deterministic class colours, clipping at and beyond the frame edges, input frame never modified, invalid input |
| `test_tracking.py` | `TrackedDetection` validation, ID attachment and config validation (pure Python); the real Ultralytics ByteTrack on hand-made moving boxes: stable IDs, frames without detections, briefly missed objects, IDs reassigned after `track_buffer`, low-score continuation, reset, custom/invalid configs, errors not swallowed; missing `lap`/Ultralytics |
| `test_export.py` | `export_onnx()` with a fake Ultralytics: exact export arguments (CPU, ONNX), nothing written beside the weights, temp folder removed, missing/invalid source, invalid options and output paths, overwrite protection, a failed export keeps the previous file, export exceptions, missing or empty result, load failure, non-detection model, dependency checks (missing/too old `onnx`/`onnxslim`, no ONNX Runtime, `--no-simplify`, missing PyTorch/Ultralytics), import without optional packages; plus the opt-in real export |
| `test_onnx_validation.py` | `validate_onnx()`: missing/empty files, missing `onnx`, checker failure and success with a fake `onnx`; real ONNX files built in the test (valid, invalid graph, malformed bytes) when `onnx` is installed; YOLO layout checks (sizes, channels, types, ranks, classes, task, dynamic dimensions); class names parsed safely |
| `test_onnx_backend.py` | Provider selection (CPU, auto, CUDA requested but missing or not activated); letterbox geometry, BGR->RGB, dtype, range and padding, pixel-identical to Ultralytics' `LetterBox`; NMS (overlap, exclusive threshold, empty and zero-area boxes); raw-output decoding (best class, strict threshold, class filter, per-class NMS, 300-box limit) matching Ultralytics' `non_max_suppression` on random outputs; NMS-included outputs (no second NMS); box restoration and round trip; `OnnxYoloDetector` with a fake session: missing ONNX Runtime/OpenCV, invalid paths, load failures, unsupported inputs/outputs/metadata, input-size mismatch, dynamic models, one session reused, end-to-end detections, runtime errors; plus the opt-in real test |
| `test_backend_selection.py` | `create_detector` and `--backend` in both commands: model/backend mismatches (exit 2), ONNX image and video runs with tracking, JSON and annotated video, missing ONNX model, CUDA unavailable, PyTorch default and PyTorch JSON layout unchanged |
| `test_benchmark_stats.py` | Percentiles (hand-computed values and NumPy's default method), mean, median, min/max, sample stdev (`None` for one sample), full precision, empty or non-finite samples rejected, throughput |
| `test_benchmark_runner.py` | With a fake clock and fake detectors: initialisation, first inference and warm-up kept out of the statistics, exact latency samples and throughput, frame rotation, one detector per run (closed even on errors), missing stage timings marked unavailable, zero warm-up noted; pipeline scope with separate decode/detect/track/frame times, video ending early or too short; plan validation; model input shapes (checked against Ultralytics' `LetterBox`); library import timing |
| `test_benchmark_report.py` | Schema and key order, pooled and per-run summaries, JSON round trip at full precision, atomic writing and overwrite protection, invalid files rejected, relative paths and hashes, no personal paths in the environment, ONNX benchmarks not importing PyTorch, text summary, comparison listing differences without ranking |
| `test_benchmark_scripts.py` | `benchmark.py` and `compare_benchmarks.py`: `--help` runs nothing, invalid arguments (exit 2), missing files (exit 1, no downloads), existing output kept, complete mocked ONNX and PyTorch results (square vs minimal padding), image input, pipeline scope with tracking, failures and Ctrl+C write nothing, comparison; plus the opt-in real benchmark |
| `test_export_scripts.py` | `scripts/export_onnx.py` and `scripts/inspect_onnx.py`: `--help` creates no files, invalid arguments, config defaults and overrides, `--no-validate`, error messages and exit codes for every failure, report contents |
| `test_writer.py` | Mocked `cv2.VideoWriter`: invalid fps/size/codec/path, open failure, wrong frame size, odd sizes, ordering, release and `.partial` cleanup on success and on error, overwrite protection, FPS selection; plus a real write-and-reopen check |

## Planned coverage

- Systematic PyTorch/ONNX output comparison with tolerances

Tests that run a real pretrained model will be kept separate and opt-in.
