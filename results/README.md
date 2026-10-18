# results/

Benchmark outputs, annotated videos, and plots. **Generated outputs are not committed.** Curated, verified results will be summarised in [`docs/experiment_log.md`](../docs/experiment_log.md) together with their configuration and hardware.

No benchmark results are committed; result files stay on the machine that produced them.

Annotated videos from `detect_video.py --output-video results/<name>.mp4` are also written here by convention. They use codec `mp4v`, the source's size and FPS, and are written as `<name>.partial.mp4` until complete. Like everything else in this folder, they are ignored by Git.

## Video detection results (`detect_video.py --json`)

`python scripts/detect_video.py <video> --json results/<name>.json` writes one JSON file per run. It is written to `<name>.json.partial` first and renamed only when complete, so a file named `.json` is always a finished run. A leftover `.partial` file comes from a crash and can be deleted. An existing file is replaced only with `--overwrite`.

Schema `edgevision.video_detections`, version 1:

```json
{
  "schema": "edgevision.video_detections",
  "schema_version": 1,
  "edgevision_version": "0.1.0.dev0",
  "created_utc": "2026-10-17T11:00:00Z",
  "video": {"path": "data/input/sample.mp4", "width": 1024, "height": 576,
            "fps": 30.0, "frame_count_reported": 113, "fourcc": "h264"},
  "detector": {"backend": "pytorch", "model": "models/yolov8n.pt", "device": "cpu",
               "settings": {"input_size": 640, "conf_threshold": 0.25, "iou_threshold": 0.7, "classes": null}},
  "bbox_format": "xyxy_pixels",
  "frames": [
    {"frame_index": 0, "detections": [
      {"bbox_xyxy": [195.9, 0.3, 285.9, 149.3], "confidence": 0.855, "class_id": 0, "class_name": "person"}
    ]},
    {"frame_index": 1, "detections": []}
  ],
  "summary": {"frames_processed": 2, "frames_with_detections": 1, "frames_without_detections": 1,
              "total_detections": 1, "detections_by_class": {"person": 1},
              "max_frames": 2, "completed": true, "stop_reason": "max_frames",
              "ended_before_reported_count": false}
}
```

This example is a two-frame run (`--max-frames 2`); the values illustrate the format and are not measurements.

| Field | Notes |
|-------|-------|
| `video` | Metadata *reported* by the file; any value may be `null` if unknown |
| `detector.backend` | `pytorch` or `onnx` (`--backend`) |
| `detector.device` | `cpu` or `cuda:0`: where inference actually ran |
| `detector.execution_providers` | **ONNX backend only:** the ONNX Runtime execution providers the session actually used, in priority order, e.g. `["CPUExecutionProvider"]`. Absent for PyTorch, whose files are unchanged |
| `detector.settings` | Exactly the settings used, so the run can be repeated |
| `bbox_format` | `xyxy_pixels`: `[x1, y1, x2, y2]`, top-left and bottom-right corners in pixels of the original frame |
| `frames` | One entry per processed frame, in order, with zero-based `frame_index`; frames without detections have an empty list |
| `confidence` | Post-NMS score in `[0, 1]`; it is **not** an accuracy measure |
| `summary` | Counts are consistent: frames with + without detections = frames processed; per-class counts sum to `total_detections` |

The file contains no image data and no timing information.

### With tracking (`--tracking`)

Tracking extends the same schema (still version 1) with extra keys only; files from detection-only runs are unchanged:

```json
  "tracker": {"name": "bytetrack", "config": "ultralytics:bytetrack.yaml",
              "settings": {"tracker_type": "bytetrack", "track_high_thresh": 0.25, "track_low_thresh": 0.1,
                           "new_track_thresh": 0.25, "track_buffer": 30, "match_thresh": 0.8, "fuse_score": true}},
  "frames": [
    {"frame_index": 0, "detections": [
      {"bbox_xyxy": [...], "confidence": 0.85, "class_id": 0, "class_name": "person", "track_id": 1},
      {"bbox_xyxy": [...], "confidence": 0.31, "class_id": 16, "class_name": "dog", "track_id": null}
    ]}
  ],
  "summary": {..., "tracking": {"tracker": "bytetrack", "detections_with_track_id": 344,
                                "distinct_track_ids_observed": 18}}
```

| Field | Notes |
|-------|-------|
| `tracker` | Tracker name, where its settings came from, and the exact settings (after `bbox_format`) |
| `track_id` | Integer track ID, or `null` for a detection the tracker has not assigned to a confirmed track. IDs are unique within one run only, may have gaps, and are not persistent identities |
| `summary.tracking.distinct_track_ids_observed` | Distinct IDs that appeared in this run, **not** a count of real-world objects (lost objects can return with a new ID) |

## Benchmark results (`scripts/benchmark.py`)

`python scripts/benchmark.py ...` writes one JSON file per invocation, by default `results/benchmarks/<backend>_<device>_<scope>_<date-time>.json`. Like the other outputs, it is written as `<name>.json.partial` and renamed when complete, and an existing file is replaced only with `--overwrite`. The timing boundaries are explained in [`docs/benchmarking.md`](../docs/benchmarking.md).

Schema `edgevision.benchmark`, version 1. The structure is shown below; `…` stands for values, which depend entirely on the machine and run:

```text
{
  "schema": "edgevision.benchmark", "schema_version": 1,
  "edgevision_version": "0.1.0.dev0", "created_utc": "2026-10-18T08:00:00Z",
  "git": {"commit": "…", "uncommitted_changes": false},           # null if git is unavailable
  "benchmark": {"scope": "detector", "warmup_iterations": 20, "measured_iterations": 300, "runs": 3,
                "preload_frames": 32, "tracking": false, "batch_size": 1},
  "backend": {"name": "onnx", "runtime": "ONNX Runtime 1.31.0, providers: CPUExecutionProvider",
              "model": {"path": "models/yolov8n.onnx", "size_bytes": …, "sha256": "…"},
              "requested_device": "cpu", "device": "cpu", "execution_providers": ["CPUExecutionProvider"],
              "letterbox": "square", "model_input_shape": [1, 3, 640, 640], "precision": "fp32",
              "settings": {"input_size": 640, "conf_threshold": 0.25, "iou_threshold": 0.7, "classes": null}},
  "input": {"kind": "video", "file": {"path": "data/input/sample.mp4", "size_bytes": …, "sha256": "…"},
            "width": 1024, "height": 576, "fps": 30.0, "frame_count_reported": 113, "frames_in_memory": 32},
  "environment": {"os": {…}, "python": {"version", "implementation"}, "cpu": {"model", "architecture", "logical_cores"},
                  "nvidia_gpus": […], "packages": {…}, "onnxruntime": {…} or "torch": {…}},
  "library_import": {"modules": ["onnxruntime"], "seconds": …, "already_imported": false, "error": null},
  "summary": {"runs": 3, "measured_iterations_total": 900,
              "latency_ms_pooled": {"detect": {statistics}, "preprocess_ms": {…}, "inference_ms": {…}, "postprocess_ms": {…}},
              "detect_median_ms_per_run": […], "throughput_fps_per_run": […], "throughput_fps": {"mean", "min", "max"},
              "init_s_per_run": […], "first_inference_ms_per_run": […], "detections_total_per_run": […]},
  "runs": [{"run": 1, "init_s": …, "tracker_init_s": null, "first_inference_ms": …, "warmup_iterations": 20,
            "warmup_s": …, "measured_iterations": 300, "measured_wall_s": …, "throughput_fps": …,
            "detections_total": …, "device": "cpu", "execution_providers": [...],
            "latency_ms": {"detect": {statistics}, …}, "samples_ms": {"detect": [300 values], …}, "notes": []}, …],
  "timing_definitions": {…},  "limitations": […],  "warnings": […]
}
```

`{statistics}` is `{"count", "mean", "median", "p90", "p95", "min", "max", "stdev"}` in milliseconds.

| Field | Notes |
|-------|-------|
| `backend.model_input_shape` | The tensor the network actually received. Compare it between results: different shapes mean different amounts of work |
| `backend.device`, `execution_providers` | Where inference really ran (providers only for ONNX Runtime). `requested_device` is what was asked for |
| `input.file.sha256` | Identifies the input exactly; two results on the same input have the same hash |
| `latency_ms` / `samples_ms` keys | Detector scope: `detect` and, if the backend reports them, `preprocess_ms`, `inference_ms`, `postprocess_ms`. Pipeline scope: `decode`, `detect`, `track` (with `--tracking`), `frame`, plus the stages |
| `runs[].suspect_iterations` | Iterations more than 10× the run's median and over 1 s, e.g. because the computer went to sleep. Listed as `{iteration, ms, times_median}`; the samples themselves are unchanged. Any entry also adds a `WARNING` to `warnings`; rerun such a benchmark |
| `runs[].notes` | Facts about that run, e.g. "stage breakdown unavailable" or "the video ended after N of M measured frames" |
| `warnings` | Things that limit the result, e.g. few iterations, CPU chosen by `auto`, PyTorch minimal padding |
| `timing_definitions` | The exact meaning of every metric, stored in each file |

Statistics never use 0 as a placeholder: a value that cannot be computed is `null` (e.g. `stdev` with one sample) or absent with a note.

`python scripts/compare_benchmarks.py A.json B.json` prints results side by side and lists every setting that differs. A future step will add plots from these files.
