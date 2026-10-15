# tests/

Automated tests (pytest). **No tests exist yet.**

## Planned coverage

- Configuration parsing and validation
- Video input handling, including missing and corrupt files
- Detection output shape and coordinate conventions
- ONNX model loading and PyTorch/ONNX output comparison
- Latency aggregation, percentiles, and FPS calculations
- Empty-detection handling
- Tracker integration
- Annotated output video creation
- Benchmark result serialization

## Constraints

- The default test suite must run on CPU only, without network access or dataset downloads.
- Unit tests will use deterministic synthetic inputs and mocked inference where appropriate.
- Tests that execute a real pretrained model will be separate and opt-in.
