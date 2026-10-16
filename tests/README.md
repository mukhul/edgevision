# tests/

Automated tests (pytest). Run them from the repository root after `python -m pip install -e ".[dev]"`:

```bash
python -m pytest            # whole suite
python -m pytest -v         # one line per test
python -m pytest tests/test_config.py
```

The suite needs no GPU, no network access, no downloaded models, and none of the optional ML packages. PyTorch, ONNX Runtime and `nvidia-smi` are replaced with small fakes where a test needs them.

## Current coverage

| File | What it checks |
|------|----------------|
| `test_package.py` | The package is installed and imported from `src/` |
| `test_config.py` | Valid configs load; missing files, malformed YAML, duplicate keys, missing/unknown fields, wrong types and out-of-range values are rejected with clear messages |
| `test_device.py` | The environment report runs without optional packages; CUDA/provider detection, broken installs, `nvidia-smi` parsing and failures, and setup warnings |

## Planned coverage

- Video input handling, including missing and corrupt files
- Detection output shape and coordinate conventions
- ONNX model loading and PyTorch/ONNX output comparison
- Latency aggregation, percentiles, and FPS calculations
- Empty-detection handling
- Tracker integration
- Annotated output video creation
- Benchmark result serialization

Tests that run a real pretrained model will be kept separate and opt-in.
