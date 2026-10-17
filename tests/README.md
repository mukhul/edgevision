# tests/

Automated tests (pytest). Run them from the repository root after `python -m pip install -e ".[dev]"`:

```bash
python -m pytest            # whole suite
python -m pytest -v         # one line per test
python -m pytest tests/test_config.py
```

The suite needs no GPU, no network access, no downloaded models, and none of the optional ML packages. PyTorch, ONNX Runtime, OpenCV, Ultralytics and `nvidia-smi` are replaced with small fakes where a test needs them (the shared fake YOLO model is in `conftest.py`).

A few **integration tests** use real OpenCV to write and read back a tiny synthetic video. Install the `vision` extra (`python -m pip install -e ".[dev,vision]"`) to run them; without it they are reported as *skipped* with the reason, and every other test still runs.

**Mocked tests are not real inference.** They check EdgeVision's own code around the model. Two opt-in tests (marker `model`) run the genuine pretrained YOLOv8n on CPU: one on `bus.jpg`, which ships inside the Ultralytics package, and one on a short video built from it. They need the `detector` extra and an existing `models/yolov8n.pt` (they never download):

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
| `test_pipeline.py` | Frame order and indices reaching the detector, aggregation and summary consistency, frames/videos without detections, empty video, `max_frames`, detector failures with context, contract checks, video released on success/failure/early stop; integration with the real `VideoReader`; opt-in real-model video test |
| `test_results.py` | JSON schema, parseable output, `.partial`-then-rename behaviour, no file after errors, overwrite protection, invalid values rejected |
| `test_detect_video.py` | `scripts/detect_video.py`: arguments, config defaults vs `--max-frames`/`--all-frames`, summary, JSON contents, overwrite protection, inference failure and Ctrl+C leave no results file |

## Planned coverage

- ONNX model loading and PyTorch/ONNX output comparison
- Latency aggregation, percentiles, and FPS calculations
- Tracker integration
- Annotated output video creation
- Benchmark result serialization

Tests that run a real pretrained model will be kept separate and opt-in.
