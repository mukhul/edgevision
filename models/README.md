# models/

Locally downloaded weights and exported models. **Model files are not committed** (`*.pt`, `*.onnx`, `*.engine` are ignored).

## Contents

| File | How it is obtained |
|------|--------------------|
| `yolov8n.pt` | Pretrained Ultralytics YOLOv8n weights (~6 MB, 80 COCO classes). Downloaded by Ultralytics the first time a detector loads `models/yolov8n.pt`, for example by running `python scripts/detect_image.py <image>`. Nothing is downloaded at install time or by the tests. |
| `yolov8n.onnx` | Generated from `yolov8n.pt` by `python scripts/export_onnx.py` (needs the `detector` and `export` extras). About 12.3 MiB. |

To use other weights, pass `--model path/to/weights.pt` to `detect_image.py`, or change `model.weights` in the config. They must be an Ultralytics **detection** model; segmentation, pose or classification models are rejected.

## The ONNX export

`scripts/export_onnx.py` exports on the CPU with Ultralytics' default ONNX settings: FP32, batch 1, a fixed 640×640 input (`--imgsz`, or `--dynamic` for any size), opset chosen by Ultralytics (18 with PyTorch 2.14; `--opset` to fix it), graph simplified with onnxslim, and **no NMS inside the graph**. The result, as checked on Linux:

| | Name | Type | Shape |
|---|------|------|-------|
| Input | `images` | FLOAT | `[1, 3, 640, 640]`: RGB, values 0–1, letterboxed |
| Output | `output0` | FLOAT | `[1, 84, 8400]`: per candidate box, centre x, centre y, width, height (input pixels) and 80 class scores |

Ultralytics also stores metadata in the file (task, input size, class names, version), which `python scripts/inspect_onnx.py models/yolov8n.onnx` displays.

The file is only checked **structurally** (official ONNX checker, input/output layout). Numerical comparison with PyTorch, and the tolerances it will use, will be documented here when that step exists. The input description above is what Ultralytics' YOLO models expect; it has not yet been verified with ONNX Runtime in this project.

## License note

Ultralytics YOLO code and pretrained weights are distributed under the AGPL-3.0 license (an enterprise license is also offered). Review those terms before redistributing weights, exported models, or derived work.
