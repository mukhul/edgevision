# results/

Benchmark outputs, annotated videos, and plots. **Generated outputs are not committed.** Curated, verified results will be summarised in [`docs/experiment_log.md`](../docs/experiment_log.md) together with their configuration and hardware.

No benchmark results exist yet.

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

## Planned layout

Each run will write to its own uniquely named folder so earlier results are never silently overwritten:

```text
results/
└── <experiment_id>/
    ├── config.yaml        # exact configuration used
    ├── environment.json   # hardware, OS, library versions, execution providers, git commit
    ├── results.json       # per-run statistics and metadata
    ├── results.csv        # tabular view of the same results
    └── plots/             # figures generated from the saved results
```
