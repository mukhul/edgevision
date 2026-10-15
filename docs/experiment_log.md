# Experiment log

> **No experiments have been executed yet.** Entries will be added only for experiments that have actually been run, with their raw results kept under `results/<experiment_id>/`.

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
| — | — | — | — | — | — | — | — | — | — |

## Interpretation

To be written after results exist.
