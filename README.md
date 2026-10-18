# EdgeVision — GPU-Accelerated Real-Time Perception Engine

![Status](https://img.shields.io/badge/status-early%20development-orange)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Backends](https://img.shields.io/badge/backends-PyTorch%20%7C%20ONNX%20Runtime-lightgrey)

> [!IMPORTANT]
> **This repository is in early development.**
> Implemented so far: an installable package, validated configuration loading, an environment report, a frame-by-frame video reader, object detection with a pretrained YOLOv8n model on PyTorch, a pipeline that runs detection over every frame of a prerecorded video with a summary and optional JSON output, an annotated copy of the video with boxes, class names and confidences, optional multi-object tracking with ByteTrack (track IDs in the video and JSON), and export of the model to ONNX with structural validation of the exported file. ONNX Runtime inference, PyTorch-vs-ONNX numerical comparison, GPU execution, and benchmarking are not implemented. **No benchmark results have been recorded.** Everything described below as *planned* is the design target, not existing functionality.

## Overview

EdgeVision is an ongoing computer vision and performance-engineering project focused on building a video perception pipeline and evaluating inference performance across PyTorch and ONNX Runtime. The project aims to investigate the trade-offs between inference latency, throughput, and detection quality under reproducible benchmarking conditions.

### Motivation

Choosing an inference runtime is an engineering decision that should rest on measurements, yet runtime comparisons are easy to get wrong: warm-up iterations leak into averages, asynchronous CUDA execution is timed without synchronization, end-to-end pipeline speed is reported as pure model speed, and single best-case runs stand in for typical behaviour. EdgeVision is being built to answer a narrow question carefully — *for the same model, weights, and frames, how do PyTorch and ONNX Runtime compare on latency, throughput, and output agreement, on CPU and (where available) GPU?* — with a methodology that makes every result traceable to its configuration and hardware.

## Project status

| Capability | Planned | Implemented | Benchmarked |
|------------|:-------:|:-----------:|:-----------:|
| Installable package (`pip install -e ".[dev]"`) | ✅ | ✅ | n/a |
| Configuration loading and validation | ✅ | ✅ | n/a |
| Hardware / software environment report | ✅ | ✅ | n/a |
| Video input (OpenCV, prerecorded files) | ✅ | ✅ | n/a |
| Object detection — PyTorch backend (single image / frame) | ✅ | ✅ ¹ | — |
| Detection on video frames (pipeline, summary, JSON export) | ✅ | ✅ ² | — |
| ONNX export and structural validation | ✅ | ✅ ⁵ | n/a |
| ONNX numerical validation (PyTorch vs ONNX outputs) | ✅ | — | — |
| Object detection — ONNX Runtime CPU | ✅ | — | — |
| Object detection — ONNX Runtime CUDA (when available) | ✅ | — | — |
| Multi-object tracking (ByteTrack, optional) | ✅ | ✅ ⁴ | n/a |
| Annotated output video (boxes, class names, confidences) | ✅ | ✅ ³ | n/a |
| Latency / throughput benchmark runner | ✅ | — | — |
| Detection-quality evaluation (ground truth) | ✅ | — | — |
| Tests for the pipeline, backends, and benchmarks | ✅ | — | n/a |

- **Planned** — described in the project specification; not yet built.
- **Implemented** — exists in this repository and is covered by passing tests.
- **Benchmarked** — measured, with results and configuration recorded in [`docs/experiment_log.md`](docs/experiment_log.md).

¹ Covered by mocked unit tests **and** by a real-model test (`yolov8n.pt` on CPU, Linux, PyTorch 2.14.1 CPU build, Ultralytics 8.4.178). The real test is opt-in; see [Tests](#tests). No speed or accuracy figures have been measured.

² Covered by mocked unit tests, by integration tests with real OpenCV, and by an opt-in real-model test on a short video. The command was also run end to end with the real YOLOv8n model on a 113-frame, 1024×576 H.264 clip on Linux (CPU). Detection counts describe what the model output, not whether it was correct: the processing summary is neither an accuracy nor a speed measurement.

³ Covered by annotator tests on real pixels, mocked writer tests, real write-and-reopen integration tests, and command tests. The command was run end to end with the real YOLOv8n model on the same 113-frame clip (Linux, CPU): the annotated MP4 reopened at the source's 1024×576 and 30 FPS with all 113 frames, and sample frames were inspected visually.

⁴ Covered by tests of the real Ultralytics ByteTrack on hand-made moving boxes (stable IDs, briefly missed objects, reset, low-score continuation), mocked pipeline and command tests, and an opt-in real YOLOv8n + ByteTrack test. It was also run end to end on the 113-frame clip (Linux, CPU) with an annotated video and JSON, and sample frames were inspected visually. No tracking accuracy has been measured: there is no ground truth.

⁵ Covered by mocked unit tests (Ultralytics and `onnx` replaced by fakes), tests on small real ONNX files built with the `onnx` package, and an opt-in real-model test that exports `yolov8n.pt` at 320×320 and validates it. The command was also run for real on Linux (CPU; PyTorch 2.14.1, Ultralytics 8.4.178, onnx 1.23.2): `models/yolov8n.onnx` (12,851,106 bytes, opset 18) passed the official ONNX checker with input `images` FLOAT `[1, 3, 640, 640]` and output `output0` FLOAT `[1, 84, 8400]`. This shows the file is well-formed, **not** that it gives the same detections as PyTorch; that comparison is a later step.

**What exists today:**

- An installable `edgevision` package (src layout) whose only required dependency is PyYAML.
- `edgevision.config`: loads `configs/default.yaml` (or any given file) into typed, immutable settings. Missing fields, wrong types, out-of-range values, unknown fields and duplicate keys are rejected, and every problem is listed with its location.
- `edgevision.device`: reports Python, OS, CPU, NVIDIA GPU hardware (via `nvidia-smi`), optional packages, PyTorch CUDA support and ONNX Runtime execution providers. It works when PyTorch and ONNX Runtime are not installed, and keeps "a GPU is present" separate from "a framework can use it".
- `edgevision.video.VideoReader`: reads a prerecorded video one frame at a time (BGR, zero-based indices), with an optional `max_frames` limit, defensive metadata handling (width, height, FPS, reported frame count, codec) and guaranteed release of the OpenCV capture. OpenCV is an optional extra (`vision`).
- `scripts/inspect_video.py`: reports a video's metadata, how many frames could actually be decoded, and why reading stopped.
- `edgevision.models`: a backend-independent `Detection` record (frame index, `xyxy` box in original-image pixels, confidence, class ID, class name; validated and immutable), shared `DetectionSettings` (input size, thresholds, class filter), a `Detector` protocol, and `PytorchYoloDetector`, which runs a pretrained Ultralytics YOLO model on one BGR frame with `auto` / `cpu` / `cuda` device selection. An explicit `cuda` request fails clearly instead of silently using the CPU.
- `scripts/detect_image.py`: runs the detector on one image, prints the detections, and can save them as JSON.
- `edgevision.pipeline`: `VideoDetectionPipeline` feeds every frame from `VideoReader` to an already-created detector, one frame at a time, yields per-frame results (`FrameDetections`), and keeps only counters for its `PipelineSummary`. `edgevision.results` streams those results to a JSON file that only appears once it is complete.
- `edgevision.video.annotator`: draws each detection (box, class name, confidence) on a **copy** of a frame, in a fixed colour per class.
- `edgevision.video.writer`: writes frames to an `.mp4` file with validated size and frame rate, and publishes it only when complete.
- `edgevision.tracking`: `TrackedDetection` (a `Detection` plus `track_id`), a `Tracker` protocol, and `ByteTrackTracker`, which feeds each frame's detections to Ultralytics' ByteTrack and attaches the resulting IDs to the original detections.
- `scripts/detect_video.py`: runs detection over a video, prints a processing summary, and can save per-frame detections as JSON and/or an annotated copy of the video. `--tracking` adds ByteTrack track IDs to both.
- `edgevision.models.export`: `export_onnx()` exports a YOLO `.pt` model to ONNX on the CPU through Ultralytics' export API, without downloading anything, writing next to the weights, or replacing an existing file unless asked.
- `edgevision.models.onnx_validation`: `validate_onnx()` runs the official ONNX checker and describes the model's inputs, outputs, opset and metadata; `yolo_layout_problems()` checks that they look like a YOLO detection model.
- `scripts/export_onnx.py` (export, then validate) and `scripts/inspect_onnx.py` (validate and describe an existing `.onnx` file).
- pytest tests for all of the above. They need no GPU, network, or ML packages; OpenCV, PyTorch, Ultralytics and ONNX are replaced by fakes in unit tests. A few integration tests use real OpenCV or ONNX when installed, and opt-in tests run the real YOLOv8n model.
- A `.gitignore` that keeps weights, videos and results out of Git, and design documents for the architecture and benchmarking methodology.

## Intended scope

- Lightweight pretrained object detection (initially YOLOv8n) on recorded video
- Multi-object tracking with persistent track IDs
- PyTorch and ONNX Runtime inference backends driven through one interface
- CPU as a first-class mode, with optional NVIDIA GPU execution when hardware is actually available
- Repeatable latency and throughput benchmarking with machine-readable results (JSON/CSV) and plots
- Detection-quality evaluation when suitable ground-truth annotations are available

**Out of scope for now:** live camera input, production deployment, and real-time guarantees on any particular hardware.

## Planned architecture

```mermaid
flowchart LR
    V[/Video file/] --> R[Video reader<br/>OpenCV]
    R --> PRE[Preprocess]
    PRE --> BE{Inference backend}
    BE --> PT[PyTorch<br/>CPU / CUDA]
    BE --> OC[ONNX Runtime<br/>CPU EP]
    BE --> OG[ONNX Runtime<br/>CUDA EP]
    PT --> POST[Postprocess + NMS]
    OC --> POST
    OG --> POST
    POST --> TR[Tracker<br/>ByteTrack]
    TR --> AN[Annotator]
    AN --> W[/Annotated video/]
    POST -.-> BM[Benchmark runner]
    TR -.-> BM
    BM --> RS[(JSON / CSV)]
    RS --> PL[Plots]
```

Detection and tracking are deliberately separate stages so that detection-only benchmarks are possible. Every backend will emit the same detection record — frame index, `xyxy` box in source-frame pixels, confidence, class ID, class name — so tracking and evaluation do not depend on which runtime produced the detections. Component responsibilities are described in [`docs/architecture.md`](docs/architecture.md).

## Intended technology stack

| Area | Tools |
|------|-------|
| Language | Python 3.11 |
| Detection model | Ultralytics YOLO (YOLOv8n to start) |
| Deep learning framework | PyTorch |
| Model interchange | ONNX |
| Inference runtime | ONNX Runtime (CPU EP; CUDA EP when available) |
| Tracking | ByteTrack (existing implementation) |
| Video I/O | OpenCV |
| Data and analysis | NumPy, Pandas |
| Visualization | Matplotlib |
| Configuration | PyYAML |
| Testing | pytest |
| Compute | Local CPU; Google Colab or Kaggle GPUs when available |

TensorRT is an optional stretch goal, not a dependency.

## Repository structure

Current contents:

```text
edgevision/
├── README.md
├── pyproject.toml           # package metadata, dependencies, pytest settings
├── .gitignore
├── configs/
│   └── default.yaml         # default experiment settings (single source of defaults)
├── src/
│   └── edgevision/
│       ├── __init__.py      # package marker and version
│       ├── config.py        # YAML loading and validation
│       ├── device.py        # environment / hardware report
│       ├── optional.py      # helpful errors for missing optional packages
│       ├── tracking/
│       │   ├── __init__.py
│       │   ├── base.py      # TrackedDetection, Tracker protocol
│       │   └── bytetrack.py # ByteTrackTracker (Ultralytics ByteTrack)
│       ├── cli.py           # option parsing shared by the detection commands
│       ├── pipeline.py      # VideoReader -> Detector, per-frame results, summary
│       ├── results.py       # streaming, atomic JSON output for video detections
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py      # Detection, DetectionSettings, Detector protocol
│       │   ├── export.py    # export_onnx(): .pt -> .onnx via Ultralytics, CPU
│       │   ├── onnx_validation.py  # validate_onnx(): ONNX checker + input/output report
│       │   └── pytorch_backend.py  # PytorchYoloDetector (Ultralytics YOLO)
│       └── video/
│           ├── __init__.py
│           ├── annotator.py # draw boxes and labels on a copy of a frame
│           ├── reader.py    # frame-by-frame video reading (OpenCV)
│           └── writer.py    # validated, all-or-nothing video file writing
├── scripts/
│   ├── README.md            # CLI entry points (implemented and planned)
│   ├── detect_image.py      # run the detector on one image
│   ├── detect_video.py      # run the detector on every frame of a video
│   ├── export_onnx.py       # export the model to ONNX and validate it
│   ├── inspect_onnx.py      # validate and describe an ONNX file
│   └── inspect_video.py     # report video metadata and decodable frames
├── tests/
│   ├── README.md            # how to run tests; planned coverage
│   ├── test_config.py
│   ├── test_device.py
│   ├── conftest.py          # fake PyTorch/Ultralytics used by detector tests
│   ├── test_detect_image.py
│   ├── test_annotator.py
│   ├── test_detect_video.py
│   ├── test_detection.py
│   ├── test_export.py
│   ├── test_export_scripts.py
│   ├── test_inspect_video.py
│   ├── test_onnx_validation.py
│   ├── test_package.py
│   ├── test_pipeline.py
│   ├── test_pytorch_backend.py
│   ├── test_results.py
│   ├── test_tracking.py
│   ├── test_writer.py
│   └── test_video_reader.py
├── docs/
│   ├── architecture.md      # planned components and data flow
│   ├── benchmarking.md      # planned timing methodology
│   └── experiment_log.md    # empty until experiments are run
├── data/
│   ├── README.md
│   └── input/               # local videos (git-ignored)
├── models/
│   └── README.md            # local weights (yolov8n.pt) and ONNX exports (git-ignored)
└── results/
    └── README.md            # benchmark outputs (git-ignored)
```

Planned modules under `src/edgevision/` (added only as they are implemented): `models/` (ONNX Runtime backend), `benchmarks/` (timing, runner, metrics, report), and `evaluation/`.

## Planned benchmark methodology

- **Two scopes, reported separately:** model-only inference, and end-to-end frame processing with the included stages (decode, preprocess, inference, postprocess, tracking, rendering) stated explicitly.
- **Phases kept apart:** model load time, warm-up time, and steady-state latency are recorded separately; load and warm-up never enter steady-state statistics.
- **Correct GPU timing:** CUDA work is synchronized before the clock is read; ONNX Runtime results state which execution provider was actually active.
- **Statistics:** mean, median, p90, p95 per run; throughput as frames ÷ measured wall-clock interval, reported separately from `1 / mean latency`.
- **Fairness:** identical weights, input resolution, thresholds, class filter, video, frame sequence, warm-up count, and frame count across backends; batch size 1 by default.
- **Repetition:** at least three runs per configuration where compute permits, with variation reported — never just the best run.
- **Quality:** precision / recall / F1 (and mAP only where the protocol supports it) against ground-truth annotations; without ground truth, only backend-agreement diagnostics, which are explicitly **not** accuracy.
- **Traceability:** every result records its config, git commit, hardware, OS, library versions, and execution provider.

Details: [`docs/benchmarking.md`](docs/benchmarking.md).

## Setup and usage

Requires Python 3.11 or newer. Run all commands from the repository root.

### Available now

Optional extras keep the base install light. Install only what you need:

| Extra | Installs | Needed for |
|-------|----------|------------|
| `dev` | pytest, NumPy | running the tests |
| `vision` | `opencv-python` | reading video and images |
| `detector` | Ultralytics, PyTorch (and torchvision via Ultralytics) | YOLO object detection |
| `tracking` | Ultralytics, `lap` | ByteTrack tracking (`--tracking`) |
| `export` | `onnx`, `onnxslim`, `onnxruntime` (CPU) | ONNX export and validation (together with `detector`) |

Windows (PowerShell):

```powershell
git clone https://github.com/mukhul/edgevision.git
cd edgevision
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,vision,detector,tracking,export]"
```

If PowerShell refuses to run `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt with `.venv\Scripts\activate.bat`. On Windows the default PyPI `torch` wheel is the CPU-only build, which is what a laptop without an NVIDIA GPU needs.

Linux / macOS:

```bash
git clone https://github.com/mukhul/edgevision.git
cd edgevision
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Linux without an NVIDIA GPU: install the much smaller CPU-only PyTorch build first
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,vision,detector,tracking,export]"
```

On Linux the default PyPI `torch` wheel is a large CUDA build; the extra line above avoids it. macOS wheels from PyPI are CPU/MPS builds. No CUDA-specific PyTorch build is pinned by this project.

> [!WARNING]
> **One OpenCV package only.** `opencv-python` and `opencv-python-headless` both install the same `cv2` module and silently overwrite each other. EdgeVision uses `opencv-python` because Ultralytics requires it. If an environment created before the detector was added still has `opencv-python-headless`, uninstall **both** and reinstall:
>
> ```bash
> python -m pip uninstall -y opencv-python-headless opencv-python
> python -m pip install -e ".[dev,vision,detector,tracking,export]"
> ```
>
> `python -m edgevision.device` warns if more than one OpenCV package is installed.

Then:

```bash
python -m edgevision.device          # hardware, packages, CUDA support, ONNX Runtime providers
python -m edgevision.device --json   # the same report as JSON
python -m edgevision.config          # validate configs/default.yaml (or pass another path)
python -m pytest                     # run the test suite

# Report a video's metadata and how many frames can be decoded
python scripts/inspect_video.py data/input/sample.mp4
python scripts/inspect_video.py data/input/sample.mp4 --max-frames 100

# Detect objects in one image (first run downloads yolov8n.pt, ~6 MB, to models/)
python scripts/detect_image.py path/to/photo.jpg
python scripts/detect_image.py path/to/photo.jpg --device cpu --conf 0.4 --json results/photo.json

# Detect objects in every frame of a video (summary only, or also per-frame JSON)
python scripts/detect_video.py data/input/sample.mp4
python scripts/detect_video.py data/input/sample.mp4 --max-frames 100 --json results/sample_detections.json
python scripts/detect_video.py data/input/sample.mp4 --output-video results/sample_annotated.mp4

# ...with multi-object tracking: track IDs in the annotated video and the JSON
python scripts/detect_video.py data/input/sample.mp4 --tracking --output-video results/sample_tracked.mp4 --json results/sample_tracks.json
```

`detect_image.py` takes its model, input size, thresholds and class filter from `configs/default.yaml`; `--model`, `--imgsz`, `--conf` and `--iou` override them. `--device` is `auto` (CUDA only if PyTorch can use it, otherwise CPU), `cpu`, or `cuda` (fails with an explanation if CUDA is unavailable). `--json` refuses to replace an existing file unless `--overwrite` is given. Exit status is 0 on success, 1 if the image, model or inference fails, and 2 for invalid arguments.

Reading video and detecting objects from Python:

```python
from edgevision.models import DetectionSettings, PytorchYoloDetector
from edgevision.video import VideoReader

with VideoReader("data/input/sample.mp4", max_frames=100) as reader:
    print(reader.metadata)            # width, height, fps, frame_count, fourcc (None if unknown)
    for frame_index, frame in reader:  # frame: BGR uint8 array, shape (height, width, 3)
        ...
print(reader.frames_read, reader.stop_reason)  # e.g. 100 StopReason.MAX_FRAMES

settings = DetectionSettings(input_size=640, conf_threshold=0.25, iou_threshold=0.7)
with PytorchYoloDetector("models/yolov8n.pt", device="auto", settings=settings) as detector:
    for d in detector.detect(frame, frame_index=0):  # any BGR uint8 frame
        print(d.class_name, round(d.confidence, 2), d.bbox_xyxy)  # box in original-image pixels
```

#### The detection interface

Every backend returns the same `Detection` records, so later stages (tracking, drawing, evaluation, PyTorch-vs-ONNX comparison) never depend on library-specific output:

| Field | Meaning |
|-------|---------|
| `frame_index` | Zero-based index of the frame the detection came from |
| `bbox_xyxy` | `(x1, y1, x2, y2)`: top-left and bottom-right corners in pixels of the **original** frame (not the resized model input) |
| `confidence` | Score in `[0, 1]` after non-maximum suppression |
| `class_id`, `class_name` | Model class index and label, e.g. `15`, `"cat"` (COCO classes for the pretrained model) |

Records are validated when created (finite numbers, confidence in range, non-negative integer IDs, `x1 <= x2` and `y1 <= y2`, non-empty name) and cannot be modified afterwards. Input frames must be `uint8` arrays of shape `(height, width, 3)` in OpenCV's **BGR** order.

`PytorchYoloDetector` validates inputs and settings, selects the device, calls Ultralytics, and converts its output. Ultralytics does the computer vision itself: letterbox resizing, BGR to RGB conversion, the forward pass, non-maximum suppression, and scaling boxes back to the original frame.

#### The video detection pipeline

```mermaid
flowchart LR
    V[/Video file/] --> R[VideoReader<br/>one BGR frame at a time]
    R -->|frame, frame_index| D[Detector.detect<br/>PytorchYoloDetector]
    D -->|list of Detection| F[FrameDetections<br/>per frame, in order]
    F --> S[PipelineSummary<br/>counts only]
    F -.->|optional| J[(JSON file<br/>streamed, atomic)]
```

`VideoDetectionPipeline` connects existing parts and adds no computer vision of its own. The reader hands each decoded frame and its zero-based index to `detector.detect(frame, frame_index=...)`. The detector is created once by the caller and reused for every frame. Each frame's detections come back tagged with that same index, and the pipeline rejects any that are not. Results are yielded frame by frame and not stored, so memory use does not grow with video length. The video is released when processing ends, fails, or is stopped early.

`detect_video.py` uses the same defaults and options as `detect_image.py` (model, `--device`, `--conf`, `--iou`, `--imgsz` from `configs/default.yaml`). It also takes the video path and frame limit from the config's `video` section: `--max-frames N` overrides the limit and `--all-frames` removes it. It prints a progress line every 25 frames (`--quiet` to hide), then a **processing summary**:

| Field | Meaning |
|-------|---------|
| Frames processed | Frames decoded and passed through the detector |
| Stopped because | End of video, or the frame limit |
| Frames with / without detections | Always add up to frames processed |
| Total detections | Sum of all detection records; equals the sum of the per-class counts |
| Detections by class | How many records each class name received |

The summary describes what the model *output*. It is not an accuracy measurement (there are no ground-truth labels yet) and not a performance benchmark (no timing is measured).

`--json PATH` saves every processed frame, including frames with no detections, plus video metadata, detector settings and the summary. The schema is `edgevision.video_detections` version 1, documented in [`results/README.md`](results/README.md). The file is streamed to `PATH.partial` and renamed to `PATH` only after the summary is written. A failed or interrupted run (Ctrl+C) therefore never leaves a file that looks complete, and an existing file is only replaced with `--overwrite`. Exit status: 0 success, 1 failure, 2 invalid arguments, 130 interrupted.

#### Annotated output video

`--output-video PATH.mp4` writes an annotated copy of every processed frame:

```mermaid
flowchart LR
    R[VideoReader] --> D[Detector] --> F[FrameDetections<br/>index, frame, detections]
    F --> A[annotate_frame<br/>draws on a copy] --> W[VideoWriter<br/>.partial.mp4 → .mp4]
    F -.-> J[(JSON)]
```

- **Drawing** (`edgevision.video.annotator`). Each box is drawn from `bbox_xyxy`: coordinates are rounded to whole pixels and clipped to the frame, and a box completely outside the frame is skipped. A filled label above the box (or just inside it at the top edge) shows `class confidence`, e.g. `cat 0.91`, in black or white for contrast. Colours come from a fixed 20-colour palette indexed by class ID, so a class always has the same colour. Line width and text size scale with the frame size. The input frame is never modified.
- **Writing** (`edgevision.video.writer`). Frames are annotated and written one at a time, nothing is buffered, using codec `mp4v` (MPEG-4 Part 2). The pip OpenCV builds have no H.264 encoder. The file is written as `NAME.partial.mp4` and renamed to `NAME.mp4` only after the last frame, so a failed or interrupted run leaves no video that looks finished. An existing file is only replaced with `--overwrite`.
- **Size.** The output uses the size of the decoded frames, which is the source size. The writer is opened on the first frame, so videos whose rotation metadata makes OpenCV rotate the frames are still written correctly. Every frame must have that size. Odd widths or heights are rejected, because OpenCV's encoders would silently crop them by one pixel.
- **Frame rate.** The source's FPS is used. If the source does not report a usable FPS, the output is written at 30 FPS with a warning that playback speed may be wrong. `--fps` sets it explicitly in either case. The summary states which was used: `source`, `override` or `fallback`.
- **Playback.** `mp4v` files play in VLC and common desktop players; some web browsers cannot play them.

`--output-video` can be combined with `--json`; the JSON is identical with or without it.

#### Multi-object tracking

Detection answers *what is where in this frame*. Tracking adds *which of these is the same object as before*. It links detections across frames and gives each linked object a **track ID**.

```mermaid
flowchart LR
    R[VideoReader] --> D[Detector<br/>runs once per frame] -->|Detections| T[ByteTrackTracker<br/>attaches IDs] -->|same detections + track_id| F[FrameDetections]
    F --> A[annotate_frame<br/>'cat #3 0.91'] --> W[VideoWriter]
    F -.-> J[(JSON with track_id)]
```

- **How ByteTrack is integrated.** `ByteTrackTracker` hands each frame's `Detection` records to Ultralytics' `BYTETracker`, through Ultralytics' public `Boxes` input format. `BYTETracker` reports, for every track, which input detection it matched, so the track ID is attached to the **original** detection: boxes and confidences stay exactly as the detector produced them. The detector is not run a second time, and tracking does not depend on which detector backend produced the detections.
- **The contract.** The tracker returns the same detections, in the same order, only adding IDs. Detections it has not assigned to a track stay ordinary detections, with `track_id: null` in JSON and no `#ID` in the video. So enabling tracking never changes detection counts. The pipeline checks this and resets the tracker at the start of every video, so IDs never carry over between videos.
- **How ByteTrack associates.** Each existing track's next position is predicted (a Kalman filter). Confident detections (≥ `track_high_thresh`) are matched to the predictions by box overlap. Weaker detections (> `track_low_thresh`) then get a second chance to continue tracks that were not matched. Unmatched tracks are kept as "lost" for `track_buffer` frames, then dropped. A new object gets an ID once it is confirmed, which after the first frame of a video means seen in two consecutive frames.
- **Enabling it.** `--tracking` (or `tracking.enabled: true` in the config; `--no-tracking` overrides that). Defaults come from Ultralytics' built-in `bytetrack.yaml`. `--tracker-config my.yaml` uses your own file in the same format. Install the `tracking` extra. Ultralytics needs the `lap` package for matching and would otherwise try to pip-install it at runtime, which EdgeVision prevents by checking for it first.
- **Output.** Annotated video labels read `person #3 0.92`, with colours still by class. JSON adds a `tracker` block, a `track_id` on every detection, and `summary.tracking`; the schema is in [`results/README.md`](results/README.md). Detection-only runs produce exactly the same JSON as before.
- **What the distinct-ID count means.** The summary reports how many distinct track IDs *appeared in this run*. It is **not** a count of real-world objects. An object that is occluded or missed for longer than `track_buffer` frames comes back with a new ID, two objects that cross can swap IDs, and ID numbers can have gaps, because candidate tracks that were never confirmed also use up numbers. On the 113-frame sample clip, 18 distinct IDs appeared, numbered from 1 up to 59. Identity is never guaranteed across occlusions or long gaps.

#### ONNX export and validation

**What ONNX is.** A trained PyTorch model is Python code plus weights, so it needs PyTorch to run. [ONNX](https://onnx.ai/) (Open Neural Network Exchange) is a file format that stores the model as a graph of standard operations (convolution, addition, sigmoid, …) together with its weights. Other runtimes, such as ONNX Runtime, TensorRT or OpenVINO, can load that file without PyTorch. Exporting to ONNX is the first step towards comparing PyTorch with ONNX Runtime. This project currently **exports and checks the file only**; it does not run inference with it yet.

```mermaid
flowchart LR
    PT[/models/yolov8n.pt/] -->|copied to a temp folder| UL[Ultralytics<br/>model.export format=onnx, CPU]
    UL -->|.partial.onnx → rename| ON[/models/yolov8n.onnx/]
    ON --> CK[ONNX checker<br/>+ input/output report]
```

Install the `export` extra **together with** `detector` (export needs PyTorch and Ultralytics):

```bat
python -m pip install -e ".[dev,vision,detector,tracking,export]"
```

Then, on Windows (Command Prompt or PowerShell, venv activated, from the repository root):

```bat
:: 1. Make sure the PyTorch weights exist (the first detection run downloads them). Export never downloads.
dir models\yolov8n.pt

:: 2. Export with the config defaults: models\yolov8n.pt -> models\yolov8n.onnx, 640x640, then validate
python scripts\export_onnx.py

:: 3. Re-check or describe the file at any time
python scripts\inspect_onnx.py models\yolov8n.onnx --imgsz 640
```

Choosing what to export (defaults come from `configs/default.yaml`):

| Option | Default (config field) | Meaning |
|--------|------------------------|---------|
| `--model PATH` | `models/yolov8n.pt` (`model.weights`) | Source `.pt` file. Must already exist. |
| `--output PATH` | `models/yolov8n.onnx` (`model.onnx_path`) | Where to write the `.onnx` file; missing folders are created. |
| `--imgsz N` | `640` (`model.input_size`) | Square input size baked into the graph; a multiple of 32. |
| `--opset N` | Ultralytics' choice (`export.opset: null`) | ONNX operator-set version. With PyTorch 2.14, Ultralytics picks 18. |
| `--dynamic` | off (`export.dynamic_shapes`) | Allow any batch size and image size instead of fixed `1×3×N×N`. |
| `--no-simplify` | simplify on | Skip onnxslim graph simplification (then only `onnx` is needed). |
| `--no-validate` | validate (`export.validate`) | Skip the structural check after export. |
| `--overwrite` | off | Replace an existing output file. Without it, export refuses. |

For example: `python scripts\export_onnx.py --imgsz 320 --output models\yolov8n_320.onnx`.

What happens during export:

- **CPU only, nothing downloaded.** The source must be a local `.pt` file; a missing file is an error, not a download. Export runs with `device="cpu"`.
- **No surprise installs.** Ultralytics pip-installs missing export packages at export time. EdgeVision checks for `onnx`, `onnxslim` and an ONNX Runtime package first and stops with an install hint instead.
- **No files next to your weights.** Ultralytics always writes `<weights>.onnx` beside the `.pt` file. EdgeVision gives it a copy of the weights in a temporary folder, so its log mentions a temporary path; the result is then copied to `PATH.partial.onnx` and renamed to the requested path. A failed export leaves no output and keeps any previous file.
- **Graph contents.** The model is exported as Ultralytics exports it by default: FP32, batch 1, and **without non-maximum suppression** inside the graph. The single output `output0` has shape `[1, 84, 8400]`: for each of 8,400 candidate boxes (from the 80×80, 40×40 and 20×20 grids at 640×640), 4 box values (centre x, centre y, width, height, in input pixels) followed by 80 class scores. Decoding and NMS will be part of the ONNX Runtime backend. At 320×320 there are 2,100 candidates.
- **Output.** `Exported: models/yolov8n.onnx (12.3 MiB, 12,851,106 bytes)`, then the validation result.

**Structural validation vs numerical parity.** These are two different questions:

| | Structural validation (implemented) | Numerical parity (not yet implemented) |
|---|---|---|
| Question | Is this a well-formed ONNX model with the expected inputs and outputs? | Does it produce the same numbers as the PyTorch model? |
| How | `onnx.checker.check_model(full_check=True)`, then read input/output names, types and shapes, and check the YOLO layout | Run both models on the same images and compare outputs within a tolerance |
| Catches | Corrupt or truncated files, invalid graphs, wrong input size, wrong number of classes | Wrong preprocessing, numerical drift, broken operators |

Passing structural validation therefore does **not** mean the ONNX model gives the same detections as PyTorch. The commands say so in their output.

`inspect_onnx.py` prints a report like this (real output from the Linux run):

```text
ONNX structural validation: PASSED (official ONNX checker, full check)
  File        models/yolov8n.onnx (12.3 MiB, 12,851,106 bytes)
  Opset       18 (IR version 8)
  Producer    pytorch 2.14.1
  Input       images: FLOAT [1, 3, 640, 640]
  Output      output0: FLOAT [1, 84, 8400]
  Metadata    task detect, imgsz [640, 640], 80 classes, Ultralytics 8.4.178
  YOLO layout as expected: (batch, 3, H, W) in, (batch, 4 + classes, candidates) out
  Note: structural checks only. Matching PyTorch results requires a numerical comparison.
```

Exit status for both commands: 0 success, 1 failure (missing model or package, existing output, export or validation failure, unexpected layout), 2 invalid arguments. `--help` only prints help; it never exports.

Common errors:

| Message | Cause and fix |
|---------|---------------|
| `Model file not found: models/yolov8n.pt` | Export never downloads. Run `python scripts\detect_image.py <any image>` once, or pass `--model`. |
| `models\yolov8n.onnx already exists; use overwrite=True (--overwrite) to replace it` | Protection against silently replacing a file. Add `--overwrite` or choose another `--output`. |
| `onnx is required to export ONNX models but is not installed` (or `onnxslim`, or `An ONNX Runtime package is required ...`) | Install the extra: `python -m pip install -e ".[export]"`. Without simplification only `onnx` is needed: `--no-simplify`. |
| `onnx 1.11.0 is installed, but Ultralytics' exporter needs onnx>=1.12.0,<2.0.0` | Upgrade by reinstalling the extra. |
| `PyTorch is required to export models ...` / `Ultralytics is required ...` | Install the `detector` extra too. |
| `input_size must be ... a multiple of 32` (exit 2) | YOLO's grids need sizes such as 320, 480, 640. |
| `... is not a valid ONNX model: ...` | The file is truncated, corrupt, or not ONNX. Export it again with `--overwrite`. |
| `YOLO layout NOT as expected` | The file is valid ONNX but not the expected detection model, e.g. exported at another size (`--imgsz`) or a segmentation model. |

**GPU note.** The `export` extra installs the CPU `onnxruntime` package, which is all export needs. Do not also install `onnxruntime-gpu` in the same environment: the two packages contain the same module and overwrite each other. GPU execution will be handled when the ONNX Runtime backend is added.

**Git.** `.onnx` files are ignored by `.gitignore` (`*.onnx`, which also covers `*.partial.onnx`), like the `.pt` weights. Do not commit them: they are 12 MB build outputs that anyone can regenerate with `export_onnx.py`. Check with `git status` that `models/yolov8n.onnx` does not appear.

#### Tests

`python -m pytest` runs unit tests in which OpenCV, PyTorch, Ultralytics and ONNX are replaced by fakes where needed; they check EdgeVision's own logic (validation, device choice, argument passing, output conversion, error handling) and never download anything. They do **not** prove that the real model works. Tests that need an optional package that is not installed are reported as *skipped*, not passed. Four additional tests run the genuine YOLOv8n model: on a photo bundled with Ultralytics, on a short video built from that photo, on that video with real ByteTrack, and a real ONNX export at 320×320 followed by validation (written to a temporary folder). They are opt-in, and need the `detector` extra (plus `tracking` and `export` for those tests) and an already-downloaded `models/yolov8n.pt` (run `detect_image.py` once):

```bash
# Linux / macOS
EDGEVISION_RUN_MODEL_TESTS=1 python -m pytest -m model -v
```

```bat
:: Windows Command Prompt
set EDGEVISION_RUN_MODEL_TESTS=1
python -m pytest -m model -v
```

#### Ultralytics notes

- Ultralytics creates a settings file on first import (for example `%APPDATA%\Ultralytics\settings.json` on Windows, `~/.config/Ultralytics/settings.json` on Linux).
- With its default setting `sync: true`, Ultralytics sends anonymous usage events for runs such as prediction and export when online. To opt out: `yolo settings sync=False`.
- Ultralytics and its pretrained weights are licensed under AGPL-3.0.

### Planned interface

> [!NOTE]
> **Forthcoming.** These commands show the intended interface and do not work yet.

```bash
# Process a video with a chosen backend (PyTorch or ONNX Runtime), with tracking and annotated output
python scripts/run_video.py --input data/input/sample.mp4 --backend pytorch --output results/annotated_pytorch.mp4

# Benchmark backends on identical frames
python scripts/benchmark.py --input data/input/sample.mp4 --backends pytorch onnx_cpu --warmup 20 --frames 300 --runs 3 --output results/benchmark

# Plot saved results
python scripts/plot_results.py --input results/benchmark --output results/plots
```

CUDA-enabled PyTorch wheels and GPU setup instructions will be documented separately once GPU execution has been verified.

## Roadmap

- [x] Initial repository scaffold and design documents
- [x] Installable package with a pytest setup
- [x] Configuration loading and validation
- [x] Hardware / software environment report
- [ ] Device and execution-provider selection for inference backends
- [x] Video reader with frame limit and robust error handling
- [x] Detection interface and PyTorch detection backend (single frame)
- [x] Run detection on video frames, with processing summary and JSON export
- [x] Annotated video output
- [x] Multi-object tracking with ByteTrack (optional, `--tracking`)
- [x] ONNX export with structural validation (ONNX checker, input/output shapes)
- [ ] Numerical agreement between ONNX and PyTorch outputs (finite values, tolerances)
- [ ] ONNX Runtime CPU backend
- [ ] Benchmark runner with JSON / CSV export
- [ ] Plot generation from saved results
- [ ] CPU-only unit and integration tests
- [ ] ONNX Runtime CUDA backend (verified execution provider)
- [ ] Detection-quality evaluation on an annotated dataset
- [ ] First recorded benchmarks (Experiments A–E in [`docs/experiment_log.md`](docs/experiment_log.md))

## Benchmark results

None yet. Results will be published in [`docs/experiment_log.md`](docs/experiment_log.md) only after they have been measured, together with the hardware, software versions, and configuration that produced them.

## Known limitations

- There is no ONNX Runtime backend or benchmarking yet. The ONNX file is checked structurally only; nobody has yet compared its outputs with PyTorch's, so its detections are unverified.
- ONNX export has been run for real on Linux only (CPU). The exported graph has no NMS, a fixed 640×640 input unless `--dynamic` is used, and FP32 weights.
- Tracking quality is not measured: there are no ground-truth track annotations, so no HOTA/MOTA/IDF1. Track IDs are only consistent within one run and are not persistent identities.
- Only ByteTrack is supported. Detections pass through the detector's confidence threshold (0.25 by default) before tracking, so ByteTrack's low-score second pass only sees detections above that; Ultralytics' own `model.track()` lowers the threshold to 0.1 for this reason. A lower `--conf` gives the tracker more to work with, but also adds more low-confidence detections to the output.
- Annotated videos use `mp4v`, which some web browsers cannot play. Source videos with an odd width or height cannot be written (OpenCV would crop them).
- Detections are not checked against ground truth, so nothing here measures accuracy. On a short cat clip, YOLOv8n labelled many objects as other classes (for example "potted plant" and "dog"), which is a reminder that detection counts are not correctness.
- Real model inference has been verified on Linux (CPU). GPU (CUDA) inference has not been tested: no NVIDIA GPU was available.
- The first use of an official model name (e.g. `yolov8n.pt`) needs an internet connection to download the weights.
- Video metadata is what the file *reports*; the frame count in particular can be missing or wrong. OpenCV signals the end of a video and an undecodable frame the same way, so a corrupt file may simply end early (the inspection command flags this when the reported count is larger).
- The `output` and `benchmarking` sections of `configs/default.yaml` are validated but not yet used; fields may still change as those stages are built.
- The environment report shows what the installed frameworks *offer*. Whether inference really runs on a GPU will only be verified once the inference backends exist.
- Free cloud GPUs vary in model and availability, so any GPU results will apply only to the recorded hardware.
- ONNX Runtime may not be faster than PyTorch for a given model and device; the project will report whatever is observed.

## Future work

- Optional TensorRT experiments (via the ONNX Runtime TensorRT execution provider or a standalone engine)
- FP16 inference, validated against the FP32 baseline
- Batch-size experiments alongside the default batch size of 1
- ONNX Runtime I/O binding to measure the cost of host ↔ device copies
- Tracking metrics (e.g. HOTA, IDF1) once annotated multi-object tracking data is included

## License

A license has not been chosen yet. Note that Ultralytics YOLO and its pretrained weights are distributed under AGPL-3.0, which should be considered when choosing a license for this project. Model weights, datasets, and videos are not distributed in this repository.
