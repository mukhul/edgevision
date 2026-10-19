# data/

Local input videos and evaluation datasets. **Nothing in this directory except this file and `input/.gitkeep` is committed.**

| Path | Purpose |
|------|---------|
| `input/` | Prerecorded video files used as pipeline and benchmark inputs |
| `eval/` (suggested) | Your labelled evaluation data for `scripts/evaluate.py`: `annotations.json` (COCO format) and `images/`. See [`docs/evaluation.md`](../docs/evaluation.md). None is included |

## Guidelines

- Use only videos and datasets you have the right to use, and respect their licenses.
- Check a new video before using it: `python scripts/inspect_video.py data/input/<file>`. It shows the resolution, FPS, codec, and how many frames actually decode.
- Record every input used in an experiment below, so results can be traced to their source.
- Benchmarks should use a fixed video and a fixed frame limit so that backends see identical frame sequences.

## Inputs used in recorded experiments

| Identifier | Source | License | Resolution | FPS | Frames | Notes |
|------------|--------|---------|------------|-----|--------|-------|
| `input/sample.mp4` | Supplied by the repository owner (not committed) | Not recorded | 1024×576, H.264 | 30 | 113 | SHA-256 `3474870b318140d5afb951fd3b8d2dff58305f36a345523e65617b591ee36a26`, 684,763 bytes. Used in Experiments A, B, D, E and for `docs/assets/demo_tracking.gif` |
