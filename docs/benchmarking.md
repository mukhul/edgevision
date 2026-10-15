# Benchmarking methodology (planned)

> **Status:** methodology only. No benchmark code exists and no measurements have been taken.

## Goals

Compare PyTorch and ONNX Runtime inference on identical inputs, under conditions that are documented, repeatable, and fair, without overstating results in either direction.

## What will be measured

| Measurement | Definition |
|-------------|------------|
| Model load time | Time to construct the model or inference session, reported separately |
| Warm-up time | Time for the configured number of warm-up iterations; excluded from steady-state statistics |
| Model-only latency | Per-frame time for the forward pass on an already-preprocessed input |
| End-to-end latency | Per-frame time across an explicitly listed set of stages (see below) |
| Throughput | Frames processed ÷ wall-clock duration of the measured interval |
| Memory | Process CPU memory and GPU memory, where measurable |

**Throughput is not `1 / mean latency`.** Both may be reported, and they will be labelled separately.

### End-to-end scope

Each end-to-end result will state which of these stages are inside the timed region:

- video decoding
- preprocessing
- inference
- postprocessing / NMS
- tracking
- rendering and video writing

## Timing rules

- Use a monotonic high-resolution clock (`time.perf_counter_ns`).
- Exclude model loading, export, and warm-up from steady-state statistics.
- **CUDA (PyTorch):** kernels launch asynchronously; synchronize the device before reading the clock at the end of a timed region, or use CUDA events.
- **CUDA (ONNX Runtime):** confirm that `CUDAExecutionProvider` is the active provider of the session, not merely installed. State whether host↔device copies are inside the timed region.
- Default batch size is 1 (streaming-style inference). Batching is an optional, separately reported experiment.

## Statistics

- Per run: mean, median, p90, p95, min, max, and sample count.
- Percentiles are computed only when there are enough samples to be meaningful; the method will be documented.
- At least 3 repeated runs where compute permits; report variation across runs. Never report only the best run.

## Fair comparison protocol

Held fixed across all backends in a comparison:

- model architecture and weights
- input resolution
- confidence and IoU thresholds, class filtering
- source video and exact frame sequence
- warm-up count and measured frame count
- hardware and power settings, where controllable

Model-only and full-pipeline benchmarks are reported separately. Measurements collected under different conditions (different machines, a shared Colab GPU vs local CPU, etc.) are not presented as directly comparable.

## Detection quality

- **Ground-truth mode:** match predictions to annotations (COCO-format preferred) with documented IoU rules; report precision, recall, F1; report mAP only if the implemented protocol supports it.
- **Inference-only mode:** numerical agreement between backends, detection counts, and confidence distributions. These are diagnostics and **do not** measure accuracy.

## Recorded with every result

Experiment ID, timestamp, git commit, model and weights identifier, input video identifier, input resolution, frames processed, backend, execution provider, precision, device/GPU name, software versions, warm-up count, number of runs, latency statistics, throughput, memory, quality metrics (if any), and any errors, warnings, or skipped measurements.

## Known limitations

- Free cloud GPUs (Colab, Kaggle) are shared and vary in model and availability; results from them are specific to the recorded hardware and session.
- Thermal throttling, background load, and power settings can affect timings and are only partly controllable.
