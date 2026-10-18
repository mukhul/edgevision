"""Structural validation of ONNX model files.

Usage::

    from edgevision.models.onnx_validation import validate_onnx, yolo_layout_problems

    info = validate_onnx("models/yolov8n.onnx")
    print(info.inputs, info.outputs, info.opset)
    problems = yolo_layout_problems(info, input_size=640)   # [] if the layout is as expected

What this checks:

* the file exists and is not empty;
* the official ONNX checker (``onnx.checker.check_model`` with
  ``full_check=True``, which includes shape inference) accepts it;
* the graph has inputs and outputs, whose names, element types and shapes are
  reported;
* optionally, that the layout matches an Ultralytics YOLO detection model.

What it does **not** check: whether the model computes the same results as
the original PyTorch model, how accurate it is, or how fast it runs. Those
need inference, which is a separate step.

Needs the ``onnx`` package (the ``export`` extra), imported only when used.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from edgevision.optional import import_optional

#: A dimension is a fixed size (int), a named symbolic size (str, e.g. "batch"), or unknown (None).
Dim = int | str | None


class OnnxValidationError(Exception):
    """The file is missing, empty, unreadable, or not a valid ONNX model."""


@dataclass(frozen=True)
class TensorInfo:
    name: str
    element_type: str  # e.g. "FLOAT"
    shape: tuple[Dim, ...]

    @property
    def shape_text(self) -> str:
        return "[" + ", ".join("?" if d is None else str(d) for d in self.shape) + "]"


@dataclass(frozen=True)
class OnnxModelInfo:
    path: Path
    size_bytes: int
    ir_version: int
    opset: int | None  # default-domain ("ai.onnx") opset version
    producer: str
    inputs: tuple[TensorInfo, ...]
    outputs: tuple[TensorInfo, ...]
    metadata: dict[str, str] = field(default_factory=dict)  # e.g. Ultralytics' task, imgsz, names

    @property
    def class_names(self) -> dict[int, str] | None:
        """Class names stored by Ultralytics in the metadata, if present and readable."""
        return parse_class_names(self.metadata.get("names"))


def parse_class_names(raw: str | None) -> dict[int, str] | None:
    """Parse the ``names`` metadata Ultralytics stores in an ONNX file.

    It is a Python dict literal such as ``"{0: 'person', 1: 'bicycle'}"``. It is
    parsed with ``ast.literal_eval``, which only accepts literals and never
    runs code. Returns None if it is missing or not a ``{int: str}`` dict.
    """
    if not isinstance(raw, str):
        return None
    try:
        names = ast.literal_eval(raw)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None
    if isinstance(names, dict) and all(
        isinstance(k, int) and not isinstance(k, bool) and isinstance(v, str) for k, v in names.items()
    ):
        return names
    return None


def _tensor_info(onnx: Any, value_info: Any) -> TensorInfo:
    tensor_type = value_info.type.tensor_type
    dims: list[Dim] = []
    for d in tensor_type.shape.dim:
        if d.HasField("dim_value"):
            dims.append(int(d.dim_value))
        elif d.HasField("dim_param") and d.dim_param:
            dims.append(str(d.dim_param))
        else:
            dims.append(None)
    try:
        element_type = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
    except ValueError:
        element_type = f"UNKNOWN({tensor_type.elem_type})"
    return TensorInfo(value_info.name, element_type, tuple(dims))


def validate_onnx(path: str | os.PathLike[str]) -> OnnxModelInfo:
    """Run the ONNX checker on ``path`` and describe the model's inputs and outputs.

    Raises:
        OnnxValidationError: the file is missing, empty, or rejected by the checker.
        MissingDependencyError: the ``onnx`` package is not installed.
    """
    path = Path(path)
    if not path.exists():
        raise OnnxValidationError(f"ONNX file not found: {path} (looked relative to {Path.cwd()})")
    if not path.is_file():
        raise OnnxValidationError(f"ONNX path is not a file: {path}")
    size = path.stat().st_size
    if size == 0:
        raise OnnxValidationError(f"ONNX file is empty: {path}")

    onnx = import_optional("onnx", display_name="onnx", extra="export", purpose="validate ONNX models")
    try:
        # Passing the path (not a loaded model) also works for models with external weight files.
        onnx.checker.check_model(str(path), full_check=True)
    except Exception as exc:
        raise OnnxValidationError(f"{path} is not a valid ONNX model: {type(exc).__name__}: {exc}") from exc
    try:
        model = onnx.load(str(path), load_external_data=False)
    except Exception as exc:
        raise OnnxValidationError(f"Could not read {path}: {type(exc).__name__}: {exc}") from exc

    graph = model.graph
    initializers = {init.name for init in graph.initializer}  # weights, not real inputs (older IR versions)
    inputs = tuple(_tensor_info(onnx, v) for v in graph.input if v.name not in initializers)
    outputs = tuple(_tensor_info(onnx, v) for v in graph.output)
    if not inputs or not outputs:
        raise OnnxValidationError(f"{path} has {len(inputs)} input(s) and {len(outputs)} output(s); expected at least one of each")

    opset = next((int(o.version) for o in model.opset_import if o.domain in ("", "ai.onnx")), None)
    producer = " ".join(p for p in (model.producer_name, model.producer_version) if p)
    return OnnxModelInfo(
        path=path,
        size_bytes=size,
        ir_version=int(model.ir_version),
        opset=opset,
        producer=producer,
        inputs=inputs,
        outputs=outputs,
        metadata={p.key: p.value for p in model.metadata_props},
    )


def yolo_layout_problems(info: OnnxModelInfo, input_size: int | None = None) -> list[str]:
    """Differences from an Ultralytics YOLO detection export without built-in NMS.

    Expected: one FLOAT input ``(batch, 3, H, W)`` and one FLOAT output
    ``(batch, 4 + classes, candidates)``. If ``input_size`` is given, fixed
    ``H``/``W`` must equal it. Symbolic (dynamic) dimensions are accepted.
    Returns an empty list when nothing is wrong.
    """
    problems: list[str] = []
    if len(info.inputs) != 1:
        problems.append(f"expected 1 input, found {len(info.inputs)}")
    if len(info.outputs) != 1:
        problems.append(f"expected 1 output, found {len(info.outputs)}")
    if problems:
        return problems

    image, output = info.inputs[0], info.outputs[0]
    if image.element_type != "FLOAT":
        problems.append(f"input {image.name!r} is {image.element_type}, expected FLOAT")
    if len(image.shape) != 4:
        problems.append(f"input {image.name!r} has shape {image.shape_text}, expected (batch, 3, height, width)")
    else:
        if isinstance(image.shape[1], int) and image.shape[1] != 3:
            problems.append(f"input {image.name!r} has {image.shape[1]} channels, expected 3")
        if input_size is not None:
            for axis, label in ((2, "height"), (3, "width")):
                if isinstance(image.shape[axis], int) and image.shape[axis] != input_size:
                    problems.append(f"input {label} is {image.shape[axis]}, expected {input_size}")
    if output.element_type != "FLOAT":
        problems.append(f"output {output.name!r} is {output.element_type}, expected FLOAT")
    if len(output.shape) != 3:
        problems.append(f"output {output.name!r} has shape {output.shape_text}, expected (batch, 4 + classes, candidates)")
    else:
        names = info.class_names
        if names is not None and isinstance(output.shape[1], int) and output.shape[1] != 4 + len(names):
            problems.append(
                f"output has {output.shape[1]} values per candidate, expected 4 box values + {len(names)} classes"
            )
    task = info.metadata.get("task")
    if task is not None and task != "detect":
        problems.append(f"metadata says task {task!r}, expected 'detect'")
    return problems
