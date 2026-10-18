"""Object detection with an exported YOLO ONNX model running on ONNX Runtime.

Usage::

    from edgevision.models import DetectionSettings, OnnxYoloDetector

    with OnnxYoloDetector("models/yolov8n.onnx", device="cpu") as detector:
        print(detector.providers)                  # e.g. ('CPUExecutionProvider',)
        detections = detector.detect(frame, frame_index=0)  # frame: BGR uint8 array

Unlike the PyTorch backend, where Ultralytics does all the computer vision,
here EdgeVision does every step itself. Each step reproduces what Ultralytics
does for an ONNX model, so both backends interpret the model the same way:

1. **Preprocess** (:func:`preprocess`): letterbox the BGR frame to the model's
   square input (resize keeping the aspect ratio, pad the rest with grey 114),
   BGR -> RGB, HWC -> CHW, uint8 0-255 -> float32 0-1, add a batch dimension.
2. **Run** the ONNX Runtime session (created once, reused for every frame).
3. **Decode** the output. Two layouts are supported:

   * ``raw`` - ``(1, 4 + classes, candidates)``, e.g. ``(1, 84, 8400)``, the
     default Ultralytics export (``scripts/export_onnx.py``). Each candidate
     is centre x, centre y, width, height (input pixels) and one score per
     class. EdgeVision keeps each candidate's best class, applies the
     confidence threshold and class filter, then non-maximum suppression.
   * ``nms`` - ``(1, max_detections, 6)``, e.g. ``(1, 300, 6)``: the model
     already ran NMS (Ultralytics ``nms=True`` or end-to-end models). Rows are
     x1, y1, x2, y2, score, class. Only the threshold and class filter are
     applied, so NMS never runs twice.

4. **Restore** the boxes to original-frame pixels (undo padding and scaling)
   and clip them to the frame.

ONNX Runtime, OpenCV and NumPy are optional dependencies, imported only when
a detector is created. Nothing is ever downloaded: the ``.onnx`` file must
already exist (create it with ``scripts/export_onnx.py``).
"""

from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

from edgevision.models.base import (
    Detection,
    DetectionSettings,
    DetectorError,
    DeviceUnavailableError,
    InferenceError,
    ModelLoadError,
    validate_bgr_frame,
    validate_frame_index,
)
from edgevision.models.onnx_validation import parse_class_names
from edgevision.models.pytorch_backend import DEVICES
from edgevision.optional import MissingDependencyError, import_optional, install_hint

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

CPU_PROVIDER = "CPUExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"

#: Same limits Ultralytics uses in non_max_suppression().
MAX_DETECTIONS = 300  # boxes kept per frame after NMS
MAX_NMS_CANDIDATES = 30_000  # highest-scoring candidates passed into NMS
CLASS_OFFSET = 7680.0  # shifts each class's boxes apart so one NMS pass works per class
PAD_VALUE = 114  # grey used for letterbox padding

OUTPUT_RAW = "raw"
OUTPUT_NMS = "nms"


# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------


def _load_onnxruntime() -> Any:
    try:
        return import_optional(
            "onnxruntime", display_name="ONNX Runtime", extra="onnxruntime", purpose="run ONNX models"
        )
    except MissingDependencyError as exc:
        raise MissingDependencyError(
            f"{exc} (or, for NVIDIA GPUs, {install_hint('onnxruntime-gpu')}; install only one of the two)"
        ) from exc.__cause__


def _load_cv2() -> Any:
    return import_optional("cv2", display_name="OpenCV", extra="vision", purpose="preprocess frames for ONNX models")


# ---------------------------------------------------------------------------
# Execution providers
# ---------------------------------------------------------------------------


def select_providers(requested: Any, available: list[str]) -> list[str]:
    """Execution providers to request for ``"auto"`` / ``"cpu"`` / ``"cuda"``, in priority order.

    ``cpu`` uses only the CPU provider. ``cuda`` requires the CUDA provider and
    never silently falls back to CPU. ``auto`` asks for CUDA when this ONNX
    Runtime build offers it, with CPU as the fallback.
    """
    if not isinstance(requested, str) or requested not in DEVICES:
        raise ValueError(f"device must be one of {list(DEVICES)}, got {requested!r}")
    if CPU_PROVIDER not in available:
        raise DeviceUnavailableError(
            f"This ONNX Runtime build offers no {CPU_PROVIDER} (available: {', '.join(available) or 'none'})."
        )
    if requested == "cpu":
        return [CPU_PROVIDER]
    if CUDA_PROVIDER in available:
        return [CUDA_PROVIDER, CPU_PROVIDER]
    if requested == "auto":
        return [CPU_PROVIDER]
    raise DeviceUnavailableError(
        "device='cuda' was requested, but the installed ONNX Runtime offers no CUDAExecutionProvider "
        f"(available: {', '.join(available)}). The 'onnxruntime' package is CPU-only. Use device='cpu' or "
        f"'auto', or on a machine with an NVIDIA GPU replace it with the GPU build: "
        f"python -m pip uninstall -y onnxruntime, then {install_hint('onnxruntime-gpu')}."
    )


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Letterbox:
    """How a frame was mapped into the model input, needed to map boxes back.

    ``gain_x``/``gain_y`` are resized size / original size; ``pad_x``/``pad_y``
    are the left and top padding in input pixels.
    """

    gain_x: float
    gain_y: float
    pad_x: int
    pad_y: int


def letterbox(frame: np.ndarray, size: int, cv2: Any) -> tuple[np.ndarray, Letterbox]:
    """Resize ``frame`` to fit a ``size`` x ``size`` square, keeping its aspect ratio, and pad with grey.

    Matches Ultralytics' ``LetterBox(auto=False, center=True)``: the image is
    centred, odd padding puts the extra pixel at the bottom/right, and
    rounding follows the same rules so the result is pixel-identical.
    """
    height, width = frame.shape[:2]
    ratio = min(size / height, size / width)
    new_w, new_h = round(width * ratio), round(height * ratio)
    pad_w, pad_h = (size - new_w) / 2, (size - new_h) / 2
    top, bottom = round(pad_h - 0.1), round(pad_h + 0.1)
    left, right = round(pad_w - 0.1), round(pad_w + 0.1)
    image = frame
    if (width, height) != (new_w, new_h):
        image = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
    return image, Letterbox(new_w / width, new_h / height, left, top)


def preprocess(frame: np.ndarray, size: int, cv2: Any) -> tuple[np.ndarray, Letterbox]:
    """BGR uint8 ``(H, W, 3)`` frame -> float32 ``(1, 3, size, size)`` RGB tensor in [0, 1]."""
    import numpy as np

    image, box = letterbox(frame, size, cv2)
    tensor = image[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, then HWC -> CHW
    tensor = np.ascontiguousarray(tensor, dtype=np.float32) / np.float32(255.0)
    return tensor[None], box  # add the batch dimension


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy non-maximum suppression; returns kept indices, highest score first.

    A box is dropped when its IoU with an already kept box is greater than
    ``iou_threshold`` (the same rule as ``torchvision.ops.nms``).
    """
    import numpy as np

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        inter_w = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        inter_h = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = inter_w * inter_h
        union = areas[i] + areas[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def decode_raw_output(
    output: np.ndarray, num_classes: int, settings: DetectionSettings
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode one image's ``(4 + classes, candidates)`` output into NMS-filtered detections.

    Returns ``(boxes_xyxy, scores, class_ids)`` in model-input pixels, highest
    score first. Mirrors Ultralytics' ``non_max_suppression`` (single label
    per box, per-class NMS, at most 300 detections).
    """
    import numpy as np

    if output.ndim != 2 or output.shape[0] != 4 + num_classes:
        raise InferenceError(
            f"Expected output rows of 4 box values + {num_classes} class scores, got shape {output.shape}"
        )
    preds = output.T.astype(np.float32, copy=False)  # (candidates, 4 + classes)
    class_scores = preds[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(len(preds)), class_ids]
    keep = scores > settings.conf_threshold  # strictly greater, as in Ultralytics
    if settings.classes is not None:
        keep &= np.isin(class_ids, settings.classes)
    xywh, scores, class_ids = preds[keep, :4], scores[keep], class_ids[keep]
    empty = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64))
    if not len(scores):
        return empty
    if len(scores) > MAX_NMS_CANDIDATES:
        top = np.argsort(-scores, kind="stable")[:MAX_NMS_CANDIDATES]
        xywh, scores, class_ids = xywh[top], scores[top], class_ids[top]

    boxes = np.empty_like(xywh)  # centre x, centre y, width, height -> x1, y1, x2, y2
    half_w, half_h = xywh[:, 2] / 2, xywh[:, 3] / 2
    boxes[:, 0], boxes[:, 1] = xywh[:, 0] - half_w, xywh[:, 1] - half_h
    boxes[:, 2], boxes[:, 3] = xywh[:, 0] + half_w, xywh[:, 1] + half_h

    offsets = (class_ids.astype(np.float32) * np.float32(CLASS_OFFSET))[:, None]
    kept = nms(boxes + offsets, scores, settings.iou_threshold)[:MAX_DETECTIONS]
    return boxes[kept], scores[kept], class_ids[kept].astype(np.int64)


def decode_nms_output(
    output: np.ndarray, settings: DetectionSettings
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode one image's ``(detections, 6)`` output from a model that already ran NMS.

    Rows are x1, y1, x2, y2, score, class in model-input pixels. Only the
    confidence threshold and class filter are applied; NMS is not repeated.
    """
    import numpy as np

    if output.ndim != 2 or output.shape[1] != 6:
        raise InferenceError(f"Expected (detections, 6) rows of x1, y1, x2, y2, score, class; got shape {output.shape}")
    rows = output.astype(np.float32, copy=False)
    class_values = rows[:, 5]
    class_ids = np.rint(class_values).astype(np.int64)
    keep = rows[:, 4] > settings.conf_threshold
    if settings.classes is not None:
        keep &= np.isin(class_ids, settings.classes)
    rows, class_ids = rows[keep], class_ids[keep]
    order = np.argsort(-rows[:, 4], kind="stable")[:MAX_DETECTIONS]
    return rows[order, :4], rows[order, 4], class_ids[order]


def restore_boxes(boxes: np.ndarray, box: Letterbox, width: int, height: int) -> np.ndarray:
    """Map ``x1, y1, x2, y2`` boxes from model-input pixels back to the original frame, clipped to it."""
    import numpy as np

    out = boxes.astype(np.float32, copy=True)
    out[:, [0, 2]] = (out[:, [0, 2]] - box.pad_x) / np.float32(box.gain_x)
    out[:, [1, 3]] = (out[:, [1, 3]] - box.pad_y) / np.float32(box.gain_y)
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, width)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, height)
    return out


# ---------------------------------------------------------------------------
# Model inspection
# ---------------------------------------------------------------------------


def _literal(text: str | None) -> Any:
    try:
        return ast.literal_eval(text) if isinstance(text, str) else None
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None


def _fixed(dim: Any) -> int | None:
    """A dimension's value if it is a fixed positive integer, else None (dynamic or unknown)."""
    return dim if isinstance(dim, int) and not isinstance(dim, bool) and dim > 0 else None


@dataclass(frozen=True)
class ModelLayout:
    """What EdgeVision needs to know about an ONNX model's inputs and outputs."""

    input_name: str
    output_name: str
    input_size: int | None  # fixed square size, or None if the height/width are dynamic
    output_format: str  # OUTPUT_RAW or OUTPUT_NMS
    class_names: dict[int, str]


def inspect_session(session: Any, path: str) -> ModelLayout:
    """Check the session's inputs, outputs and metadata; raise ModelLoadError if unsupported."""
    def fail(problem: str) -> ModelLoadError:
        return ModelLoadError(
            f"{path} is not a supported YOLO detection model: {problem}. EdgeVision supports models exported "
            "by scripts/export_onnx.py; check the file with: python scripts/inspect_onnx.py " + path
        )

    meta = dict(session.get_modelmeta().custom_metadata_map or {})
    task = meta.get("task")
    if task is not None and task != "detect":
        raise fail(f"its metadata says task {task!r}, but only 'detect' models are supported")
    class_names = parse_class_names(meta.get("names"))
    if not class_names:
        raise fail("it has no readable class names in its metadata ('names')")

    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise fail(f"expected 1 input and 1 output, found {len(inputs)} and {len(outputs)}")
    inp, out = inputs[0], outputs[0]
    if inp.type != "tensor(float)":
        raise fail(f"input {inp.name!r} is {inp.type}; only float32 inputs are supported (no FP16/INT8 yet)")
    shape = list(inp.shape or [])
    if len(shape) != 4 or _fixed(shape[1]) not in (3, None):
        raise fail(f"input {inp.name!r} has shape {shape}, expected (batch, 3, height, width)")
    if _fixed(shape[0]) not in (1, None):
        raise fail(f"input {inp.name!r} has a fixed batch size of {shape[0]}; EdgeVision runs one frame at a time")
    height, width = _fixed(shape[2]), _fixed(shape[3])
    if (height is None) != (width is None) or (height is not None and height != width):
        raise fail(f"input {inp.name!r} is {shape[2]} x {shape[3]}; only square inputs are supported")

    args = _literal(meta.get("args"))
    nms_in_graph = (isinstance(args, dict) and args.get("nms") is True) or meta.get("end2end") == "True"
    out_shape = list(out.shape or [])
    if out.type not in ("tensor(float)", "tensor(float16)") or len(out_shape) != 3:
        raise fail(f"output {out.name!r} is {out.type} with shape {out_shape}, expected a 3-D float tensor")
    num_classes = len(class_names)
    if nms_in_graph:
        if _fixed(out_shape[2]) not in (6, None):
            raise fail(f"the model has NMS built in, so (batch, detections, 6) was expected, got {out_shape}")
        output_format = OUTPUT_NMS
    elif _fixed(out_shape[1]) in (4 + num_classes, None):
        output_format = OUTPUT_RAW
    elif _fixed(out_shape[2]) == 6:
        output_format = OUTPUT_NMS
    else:
        raise fail(
            f"output shape {out_shape} matches neither (batch, 4 + {num_classes} classes, candidates) "
            "nor (batch, detections, 6)"
        )
    return ModelLayout(inp.name, out.name, height, output_format, class_names)


# ---------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------


class OnnxYoloDetector:
    """YOLO detector on ONNX Runtime, implementing the :class:`~edgevision.models.base.Detector` protocol.

    Args:
        model: Path to an existing ``.onnx`` file exported from a YOLO detection model.
        device: ``"auto"`` (default), ``"cpu"`` or ``"cuda"``; see :func:`select_providers`.
        settings: Input size, thresholds and class filter. For a model with a
            fixed input size, ``settings.input_size`` must match it.

    Attributes:
        providers: The execution providers the session actually uses, in
            priority order, as reported by ONNX Runtime after loading.
        device: ``"cuda:0"`` if the CUDA provider is active, else ``"cpu"``.
        output_format: ``"raw"`` (EdgeVision runs NMS) or ``"nms"`` (built in).

    Raises:
        MissingDependencyError: ONNX Runtime, OpenCV or NumPy is not installed.
        DeviceUnavailableError: ``device="cuda"`` but the CUDA provider cannot be used.
        ModelLoadError: the file is missing, invalid, or not a supported YOLO detection model.
    """

    backend_name = "onnx"

    def __init__(
        self,
        model: str | os.PathLike[str],
        *,
        device: str = "auto",
        settings: DetectionSettings | None = None,
    ) -> None:
        if settings is None:
            settings = DetectionSettings()
        if not isinstance(settings, DetectionSettings):
            raise TypeError(f"settings must be a DetectionSettings, got {type(settings).__name__}")
        if not isinstance(model, (str, os.PathLike)) or not str(model).strip():
            raise ValueError(f"model must be a non-empty path to an .onnx file, got {model!r}")
        if not isinstance(device, str) or device not in DEVICES:
            raise ValueError(f"device must be one of {list(DEVICES)}, got {device!r}")

        self.settings = settings
        self.model_path = os.fspath(model)
        self.requested_device = device
        self._check_file()

        ort = _load_onnxruntime()
        self._cv2 = _load_cv2()
        import_optional("numpy", display_name="NumPy", extra="vision", purpose="run ONNX models")
        self.ort_version = str(getattr(ort, "__version__", "unknown"))

        requested = select_providers(device, list(ort.get_available_providers()))
        self._session: Any = self._create_session(ort, requested)
        self.providers: tuple[str, ...] = tuple(self._session.get_providers())
        self.device = self._check_providers(requested)

        layout = inspect_session(self._session, self.model_path)
        if layout.input_size is not None and layout.input_size != settings.input_size:
            raise ModelLoadError(
                f"{self.model_path} has a fixed {layout.input_size} x {layout.input_size} input, but input size "
                f"{settings.input_size} was requested. Pass --imgsz {layout.input_size} (or set model.input_size), "
                f"or export again with --imgsz {settings.input_size}."
            )
        self.layout = layout
        self.output_format = layout.output_format
        self.class_names: dict[int, str] = dict(layout.class_names)
        if layout.output_format == OUTPUT_NMS:
            logger.info(
                "%s already contains NMS; its IoU threshold and detection limit were fixed at export time, "
                "so iou_threshold=%s is not applied", self.model_path, settings.iou_threshold,
            )
        logger.info(
            "Loaded %s with ONNX Runtime %s, providers %s (%s output, %d classes)",
            self.model_path, self.ort_version, list(self.providers), self.output_format, len(self.class_names),
        )

    # --- loading ---------------------------------------------------------------

    def _check_file(self) -> None:
        path = Path(self.model_path)
        hint = "Create it with: python scripts/export_onnx.py (ONNX models are never downloaded)."
        if not path.exists():
            raise ModelLoadError(f"ONNX model not found: {path} (looked relative to {Path.cwd()}). {hint}")
        if not path.is_file():
            raise ModelLoadError(f"ONNX model path is not a file: {path}")
        if path.suffix.lower() != ".onnx":
            raise ModelLoadError(f"Expected an .onnx file for the ONNX backend, got {path}. {hint}")
        if path.stat().st_size == 0:
            raise ModelLoadError(f"ONNX model file is empty: {path}. Export it again with --overwrite.")

    def _create_session(self, ort: Any, providers: list[str]) -> Any:
        try:
            return ort.InferenceSession(self.model_path, sess_options=ort.SessionOptions(), providers=providers)
        except Exception as exc:
            raise ModelLoadError(
                f"ONNX Runtime could not load {self.model_path}: {type(exc).__name__}: {exc}. "
                f"Check the file with: python scripts/inspect_onnx.py {self.model_path}"
            ) from exc

    def _check_providers(self, requested: list[str]) -> str:
        """Compare what was requested with what ONNX Runtime actually activated."""
        if CPU_PROVIDER not in self.providers and CUDA_PROVIDER not in self.providers:
            raise DeviceUnavailableError(
                f"ONNX Runtime activated no CPU or CUDA provider (active: {list(self.providers)})"
            )
        if CUDA_PROVIDER in requested and CUDA_PROVIDER not in self.providers:
            # The build lists CUDA, but the session could not start it (typically missing
            # CUDA/cuDNN libraries). ONNX Runtime then silently falls back to the CPU.
            if self.requested_device == "cuda":
                self._session = None
                raise DeviceUnavailableError(
                    "device='cuda' was requested and this ONNX Runtime build offers the CUDA provider, but the "
                    f"session could not activate it (active: {', '.join(self.providers)}). This usually means "
                    "the CUDA or cuDNN libraries that onnxruntime-gpu needs are missing or the wrong version. "
                    "Use device='cpu', or install the CUDA/cuDNN versions listed for your onnxruntime-gpu release."
                )
            logger.warning(
                "CUDAExecutionProvider is offered but could not be activated; running on the CPU (%s)",
                ", ".join(self.providers),
            )
        return "cuda:0" if self.providers[0] == CUDA_PROVIDER else "cpu"

    # --- inference -------------------------------------------------------------

    @property
    def description(self) -> str:
        """Short human-readable description of the runtime, for summaries."""
        return f"ONNX Runtime {self.ort_version}, providers: {', '.join(self.providers)}"

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> list[Detection]:
        """Detect objects in one BGR frame. Returns an empty list if nothing is found."""
        import numpy as np

        frame_index = validate_frame_index(frame_index)
        validate_bgr_frame(frame)
        session = self._require_open()
        layout, s = self.layout, self.settings
        height, width = frame.shape[:2]

        tensor, box = preprocess(frame, s.input_size, self._cv2)
        try:
            outputs = session.run([layout.output_name], {layout.input_name: tensor})
        except Exception as exc:
            raise InferenceError(
                f"ONNX Runtime inference failed on frame {frame_index} (shape {frame.shape}, "
                f"providers {list(self.providers)}): {type(exc).__name__}: {exc}"
            ) from exc

        output = np.asarray(outputs[0])
        if output.ndim != 3 or output.shape[0] != 1:
            raise InferenceError(f"Expected a 3-D output with batch size 1, got shape {output.shape}")
        if layout.output_format == OUTPUT_RAW:
            boxes, scores, class_ids = decode_raw_output(output[0], len(self.class_names), s)
        else:
            boxes, scores, class_ids = decode_nms_output(output[0], s)
        boxes = restore_boxes(boxes, box, width, height)
        return self._to_detections(boxes, scores, class_ids, frame_index)

    def _to_detections(
        self, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, frame_index: int
    ) -> list[Detection]:
        detections = []
        for xyxy, score, class_id in zip(boxes.tolist(), scores.tolist(), class_ids.tolist()):
            name = self.class_names.get(int(class_id))
            if name is None:
                raise InferenceError(f"Model returned class ID {class_id}, which has no class name")
            try:
                detections.append(Detection(frame_index, tuple(xyxy), float(score), int(class_id), name))
            except ValueError as exc:
                raise InferenceError(f"Model produced an invalid detection: {exc}") from exc
        return detections

    # --- resource handling -------------------------------------------------------

    def _require_open(self) -> Any:
        if self._session is None:
            raise DetectorError("This detector has been closed; create a new one.")
        return self._session

    @property
    def closed(self) -> bool:
        return self._session is None

    def close(self) -> None:
        """Release the ONNX Runtime session. Safe to call more than once."""
        self._session = None

    def __enter__(self) -> OnnxYoloDetector:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self.closed else "open"
        return f"OnnxYoloDetector({self.model_path!r}, providers={list(self.providers)!r}, {state})"
