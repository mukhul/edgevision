# Experiment log

> Entries are added only for experiments that have actually been run with `scripts/benchmark.py`. Each entry names its result file (kept locally under `results/benchmarks/`, which Git ignores) and copies the key numbers and conditions from it: CPU/GPU, power mode, OS, library versions, backend and execution provider, model input tensor, scope, warm-up × iterations × runs, median and p95 `detect` latency, throughput per run, and any warnings. See [`benchmarking.md`](benchmarking.md).

## Planned experiments

| ID | Experiment | Description |
|----|------------|-------------|
| A | Baseline PyTorch | Model-only and full-pipeline timing with the pretrained detector |
| B | ONNX Runtime CPU | Same weights, frames and settings after ONNX export and validation |
| C | ONNX Runtime GPU | Only if `CUDAExecutionProvider` is genuinely available and selected |
| D | Repeated trials | ≥3 runs per configuration; variation across runs reported |
| E | Quality and consistency | Backend output comparison; ground-truth evaluation where annotations exist |

## Results

| Experiment ID | Date | Commit | Backend / EP | Device | Input | Frames × runs | Latency (median / p95) | Throughput | Notes |
|---------------|------|--------|--------------|--------|-------|---------------|------------------------|------------|-------|
| A | 2026-10-18 10:30 UTC | `7c787b8` + uncommitted changes | PyTorch 2.14.1+cpu via Ultralytics 8.4.178 | CPU | 1×3×640×640, square padding; 32 frames of `sample.mp4` in memory; detector scope | (20 + 300) × 3 | 120.54 / 132.29 ms | 8.22, 8.46, 8.18 frames/s | `pytorch_cpu.json`; 2117 detections per run; no warnings |
| B | 2026-10-18 15:59 UTC | `7c787b8` + uncommitted changes | ONNX Runtime 1.31.0, `CPUExecutionProvider` | CPU | Same as A | (20 + 300) × 3 | 94.61 / 97.24 ms | 10.67, 10.63, 10.65 frames/s | `onnx_cpu.json`; 2117 detections per run; no warnings |

### Conditions of A and B

- **Machine:** project laptop, AMD Ryzen 7 5700U (16 logical cores), no NVIDIA GPU, Windows 11 (build 26200), Python 3.11.9, OpenCV 4.14.0.94. The power mode and whether it was plugged in were not recorded.
- **Files (SHA-256):**
  - `data/input/sample.mp4` (1024×576, 113 frames): `3474870b…`
  - `models/yolov8n.pt`: `f59b3d83…`
  - `models/yolov8n.onnx` (opset 18): `d6285f26…`
- **Settings:** confidence 0.25, NMS IoU 0.7, all classes, FP32, batch 1, library default threads (PyTorch reported 8 intra-op threads).
- **D (repeated trials):** each backend ran 3 times with a fresh model. The run medians were 122.43, 117.98 and 122.08 ms for PyTorch, and 94.93, 94.61 and 94.32 ms for ONNX Runtime.
- **Not back to back:** A and B were recorded 5.5 h apart. An earlier attempt at B was interrupted when the laptop slept (iterations lasting hours) and was replaced with `--overwrite`; only the clean rerun is listed.

### E (partial): detection agreement, no ground truth

Two `detect_video.py --json` runs on the laptop over all 113 frames of `sample.mp4` were compared with `compare_detections.py`, using default settings: PyTorch pads to 640×384, ONNX to 640×640.

- **Detections:** PyTorch 407, ONNX Runtime 428.
- **Matched pairs:** 388 (same class, IoU ≥ 0.5). Found by only one backend: 19 PyTorch and 40 ONNX Runtime.
- **Frames:** 70 of 113 matched fully.
- **Matched pairs:** median IoU 0.992, median confidence difference 0.018.

This measures agreement, not accuracy: no labelled data exists for this video.

**With identical inputs** (Linux development container, the same clip, weights and settings): `detect_video.py --pytorch-letterbox square` against `--backend onnx` gave 426 vs 426 detections, all 426 matched, in every frame. The largest box difference was 0.0003 px and the largest score difference 3×10⁻⁶. With the default PyTorch padding, the same machine gave 395 vs 426 detections and 386 matched pairs. Padding explains the differences above.

## Change after A and B: faster ONNX decoding

The ONNX decoder now finds each candidate's best class on the model's native `(classes, candidates)` layout, and only for candidates above the threshold, instead of transposing the whole output first. Its output is bit-identical: a unit test against the previous method, and all 426 detections on the sample clip. Measured on the Linux development container (shared cloud CPU), back to back, same workload (32 frames of `sample.mp4`, 20 + 300 iterations × 3 runs):

| | Before | After |
|---|---:|---:|
| `postprocess` median / p95 | 1.61 / 2.30 ms | 0.67 / 0.93 ms |
| `detect` median, pooled | 53.61 ms | 50.94 ms |
| Detections per run | 2108 | 2108 |

The `detect` difference is within the run-to-run variation of a shared machine; the postprocess stage change is the reliable signal. A and B above were measured before this change and have not been repeated.

## Interpretation

On this laptop, ONNX Runtime ran YOLOv8n on the CPU with about 22% lower median `detect` latency and about 29% higher throughput than PyTorch. The `inference` stage (80.98 vs 110.68 ms median) accounts for the difference. EdgeVision's NumPy pre- and postprocessing for ONNX is slower than Ultralytics' (12.8 vs 7.8 ms combined, medians). Run-to-run spread was small for both backends. The result applies only to this machine and these settings; GPU behaviour is unknown.
