# Evaluation and performance analysis

EdgeVision separates three questions. Each has its own command, and they are never mixed:

| Question | Command | Needs | Measures |
|----------|---------|-------|----------|
| How fast is each backend on this machine? | `scripts/plot_benchmarks.py` | benchmark JSON files from `scripts/benchmark.py` | **Performance** (latency, throughput, stage times) |
| How similar are two runs' detections? | `scripts/compare_detections.py` | two `detect_video.py --json` files of the same video | **Agreement** between outputs, *not* accuracy |
| How accurate are the detections? | `scripts/evaluate.py` | **labelled images** (ground truth) that you provide | **Accuracy**: precision, recall, F1, AP / mAP |

> [!IMPORTANT]
> **This repository contains no labelled data, so no accuracy has been measured.** The accuracy tools are implemented and tested on synthetic data only. Detection counts in benchmark files and agreement between backends are **not** accuracy: two backends can agree on a wrong detection, and a detection found by only one of them may be right or wrong.

## 1. Performance plots

```bat
python -m pip install -e ".[plots]"
python scripts\plot_benchmarks.py results\benchmarks\pytorch_cpu.json results\benchmarks\onnx_cpu.json
```

Outputs go to `results\plots\`, or `--output FOLDER`. Existing files are only replaced with `--overwrite`.

| File | Shows |
|------|-------|
| `latency.png` | Median and p95 `detect()` latency per result file, all runs pooled; each run's median as a dot |
| `throughput.png` | Mean throughput per file; each run as a dot, with the min–max range |
| `stages.png` | Mean ms per frame in preprocess, inference, postprocess and "other" (the rest of `detect()`), with every value written out |
| `latency_runs.png` | The full distribution of `detect()` latency, one box per run. No point is removed |
| `summary.md` | Every plotted value in a table, the source files and the notes |

**How to read them honestly:**
- **Context and caveats on every figure.** Each figure names the CPU, scope, input, input tensor, and runs × iterations.
- **Every difference is printed.** The figures list any setting that differs between files, files recorded far apart in time, and runs flagged as interrupted.
- **Axes start at zero.** Bars therefore show true proportions.
- **Stages use means.** Means add up across stages; medians do not.
- **A missing metric says "not reported"**, never 0.
- **Interrupted iterations are kept.** These are iterations more than 10× the run's median and over 1 s, e.g. when the laptop slept. The distribution plot says how many lie above its axis.
- **Stage timers differ by backend.** The ONNX stages are timed by EdgeVision, the PyTorch stages by Ultralytics. The boundaries are close, but not identical.
- **A few runs on one machine are not a general result.** Variation between runs is shown so you can judge it. For the fairest comparison, run both backends back to back, plugged in and awake.

**Validation.** Every file is checked before plotting. An error names the file and every missing or non-numeric field, e.g. `missing field summary.latency_ms_pooled.detect`. Up to 4 files can be plotted together.

## 2. Agreement between two runs (no labels needed)

```bat
python scripts\detect_video.py data\input\sample.mp4 --all-frames --json results\sample_pytorch.json
python scripts\detect_video.py data\input\sample.mp4 --all-frames --backend onnx --json results\sample_onnx.json
python scripts\compare_detections.py results\sample_pytorch.json results\sample_onnx.json
```

Matching works frame by frame. Two detections match when they have the same class and IoU ≥ 0.5 (`--iou` to change it), and each detection is matched at most once, highest confidence first. The report gives:
- matched pairs, and detections only in A or only in B, overall and per class;
- the share of all distinct detections that matched, and frames where everything matched;
- the IoU of matched boxes and the confidence difference of matched pairs.

`--output FILE.json` saves it (schema `edgevision.detection_agreement`). Files that cover different frames are refused; differing settings are listed.

**What it showed on the sample clip** (113 frames, CPU):
- **With `detect_video.py`'s defaults (Windows).** PyTorch made 407 detections and ONNX 428. Of the 447 distinct detections, 388 matched (86.8%); the median IoU of matched boxes was 0.992. The cause is padding: the PyTorch backend pads frames only to 640×384, while the ONNX model always needs 640×640.
- **With the same square input (Linux).** A check gave PyTorch the same square 640×640 input. The two outputs then agreed on every detection (426 of 426), with confidences within 10⁻⁵.

So on this clip the differences come from the input padding, not from the conversion to ONNX.

## 3. Accuracy against labelled data

### Providing a labelled dataset

Put the images and one COCO JSON file anywhere; `data/eval/` is a good place, and it is git-ignored:

```text
data/eval/
├── annotations.json
└── images/
    ├── street_001.jpg
    └── ...
```

`annotations.json` uses the standard COCO "instances" format, which CVAT, Label Studio, Roboflow, makesense.ai and most other labelling tools can export:

```json
{
  "images":      [{"id": 1, "file_name": "street_001.jpg", "width": 1280, "height": 720}],
  "categories":  [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}],
  "annotations": [{"id": 10, "image_id": 1, "category_id": 3, "bbox": [412, 300, 120, 80], "iscrowd": 0}]
}
```

| Rule | Detail |
|------|--------|
| Boxes | `bbox` is `[x, y, width, height]` in pixels from the top-left corner (COCO convention); width and height must be > 0 |
| Classes | Matched to the model's classes **by name** (case-insensitive), e.g. `person`, `car`, `dog`. COCO category IDs do not need to match the model's IDs. Unknown names are listed and their boxes are not evaluated |
| Image sizes | `width`/`height` must equal the real image size; otherwise the command stops, because the boxes would not line up |
| Crowd regions | `iscrowd: 1` marks a region full of objects. A detection lying inside it (at least the IoU threshold of its area) is neither right nor wrong, as in COCO |
| Unlabelled objects | Every object of an evaluated class must be labelled in every image. A missing label turns a correct detection into a false positive |

### Running the evaluation

```bat
python scripts\evaluate.py --annotations data\eval\annotations.json --images data\eval\images
python scripts\evaluate.py --annotations data\eval\annotations.json --images data\eval\images --backend onnx
python scripts\evaluate.py --annotations data\eval\annotations.json --images data\eval\images --conf 0.001
```

- **Same options as detection.** `--backend`, `--model`, `--device`, `--conf`, `--iou` and `--imgsz` work as in the detection commands.
- **Output and printout.** Results go to `results\evaluation\<backend>_<date-time>.json` (schema `edgevision.evaluation`), plus a summary table per class. Nothing is downloaded.

### What the metrics mean

| Metric | Definition |
|--------|------------|
| TP / FP / FN | Per image, detections are matched from the highest confidence down. A detection matching an unmatched labelled box of the same class at IoU ≥ 0.50 is a true positive (TP); otherwise it is a false positive (FP), unless it lies inside a crowd region. Unmatched labels are false negatives (FN) |
| Precision | TP / (TP + FP): of the detections made, the share that were right |
| Recall | TP / (TP + FN): of the labelled objects, the share that were found |
| F1 | 2 · P · R / (P + R) |
| AP@0.5 (per class) | COCO definition: the class's detections across all images are sorted by confidence, precision is made non-increasing, sampled at 101 recall points (0, 0.01 … 1) and averaged |
| mAP@0.5 | Mean AP@0.5 over the classes that have labels in the dataset |
| mAP@0.5:0.95 | Additionally averaged over IoU thresholds 0.50, 0.55 … 0.95 (COCO's main metric) |

- **Not computable is `null`, not 0.** For example: precision with no detections, or AP for a class without labels.
- **Checked against the reference code.** The AP / mAP code gives identical results to the official COCO evaluation (`pycocotools`) on random synthetic datasets, including crowd regions. This is an optional test that runs only when `pycocotools` is installed.

### What accuracy numbers do and do not establish

- **Precision, recall and F1 depend on the confidence threshold.** They use the detections returned at `--conf` (0.25 by default), so they describe the detector as it is configured.
- **AP is lower at a high confidence threshold**, because low-confidence detections are missing. Published COCO numbers use 0.001; use `--conf 0.001` to compare with them.
- **The numbers describe your dataset only.** A few dozen images give uncertain numbers. A dataset unlike your real footage says little about it.
- **Accuracy and speed are separate.** A faster backend is not more or less accurate unless an evaluation on the same labelled data shows it.
