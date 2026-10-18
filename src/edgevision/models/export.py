"""Export a YOLO model from PyTorch (.pt) to ONNX (.onnx).

Usage::

    from edgevision.models.export import export_onnx

    result = export_onnx("models/yolov8n.pt", "models/yolov8n.onnx", input_size=640)
    print(result.path, result.size_bytes)

How it works:

* The export itself is done by Ultralytics' supported ``YOLO(...).export(format="onnx")``
  API; the model is not reconstructed by hand.
* Ultralytics always writes ``<weights>.onnx`` next to the weights file and
  would silently replace an existing one there. To avoid touching anything
  next to your weights, the ``.pt`` file is copied into a temporary folder and
  exported there. The result is then copied to ``<output>.partial.onnx`` and
  renamed to the requested output, so an interrupted export never leaves a
  file that looks finished.
* Before calling Ultralytics, the packages its exporter requires are checked
  (``onnx``; plus ``onnxslim`` and an ONNX Runtime package when simplifying).
  Ultralytics would otherwise pip-install missing ones during the export.
* The export runs on the CPU, never downloads weights (the source file must
  exist), and does nothing at import time.
* The exported graph has no NMS inside (Ultralytics' default): its single
  output holds all candidate boxes, as ``(batch, 4 + classes, candidates)``,
  and NMS has to be applied by whoever runs it.

A successful export plus structural validation (see
:mod:`edgevision.models.onnx_validation`) shows the file is a well-formed ONNX
model with the expected inputs and outputs. It does **not** show that it gives
the same detections as the PyTorch model; that needs a numerical comparison.
"""

from __future__ import annotations

import logging
import numbers
import os
import shutil
import tempfile
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from edgevision.models.base import MODEL_STRIDE
from edgevision.optional import MissingDependencyError, import_optional, install_hint

logger = logging.getLogger(__name__)

#: Lowest opset Ultralytics accepts for a YOLO detection model exported without NMS.
MIN_OPSET = 9
#: ONNX Runtime distributions Ultralytics accepts interchangeably when simplifying.
ONNXRUNTIME_DISTRIBUTIONS = ("onnxruntime", "onnxruntime-gpu", "onnxruntime-qnn", "onnxruntime-vitisai")


class ExportError(Exception):
    """The model could not be exported, or the export produced no usable file."""


@dataclass(frozen=True)
class ExportResult:
    """Where the ONNX file was written and how it was produced."""

    path: Path
    size_bytes: int
    source: Path
    input_size: int
    opset_requested: int | None  # None = Ultralytics chose it
    simplify: bool
    dynamic: bool


def partial_path_for(path: Path) -> Path:
    """``models/x.onnx`` -> ``models/x.partial.onnx``."""
    return path.with_name(f"{path.stem}.partial{path.suffix}")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _installed_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def check_export_dependencies(simplify: bool) -> None:
    """Make sure every package Ultralytics' ONNX exporter will ask for is present.

    Raises :class:`MissingDependencyError` instead of letting Ultralytics
    install packages on its own at export time.
    """
    import_optional("torch", display_name="PyTorch", extra="detector", purpose="export models")
    import_optional("ultralytics", display_name="Ultralytics", extra="detector", purpose="export models")
    import_optional("packaging", display_name="packaging", extra="export", purpose="check export dependencies")
    from packaging import specifiers

    required = [("onnx", ">=1.12.0,<2.0.0")]
    if simplify:
        required.append(("onnxslim", ">=0.1.82"))
    hint = f"Install the export dependencies with: {install_hint('export')}"
    for distribution, spec in required:
        version = _installed_version(distribution)
        if version is None:
            raise MissingDependencyError(f"{distribution} is required to export ONNX models but is not installed. {hint}")
        if version not in specifiers.SpecifierSet(spec):
            raise MissingDependencyError(
                f"{distribution} {version} is installed, but Ultralytics' exporter needs {distribution}{spec}. {hint}"
            )
    if simplify and not any(_installed_version(d) for d in ONNXRUNTIME_DISTRIBUTIONS):
        raise MissingDependencyError(
            "An ONNX Runtime package is required to simplify the exported model but none is installed. "
            f"Install one with: {install_hint('onnxruntime')} (CPU) or {install_hint('onnxruntime-gpu')} "
            "(NVIDIA GPU; install only one of the two), or export with simplify=False / --no-simplify"
        )


def _validate_arguments(
    weights: Path, output: Path, input_size: Any, opset: Any, simplify: Any, dynamic: Any, overwrite: bool
) -> None:
    if not weights.exists():
        raise ExportError(
            f"Model file not found: {weights} (looked relative to {Path.cwd()}). Export never downloads "
            "weights; get them first, e.g. by running scripts/detect_image.py once."
        )
    if not weights.is_file():
        raise ExportError(f"Model path is not a file: {weights}")
    if weights.suffix.lower() != ".pt":
        raise ExportError(f"Expected a PyTorch .pt model to export, got {weights}")
    if output.suffix.lower() != ".onnx":
        raise ValueError(f"output must end in .onnx, got {output}")
    if output.is_dir():
        raise ValueError(f"output is a directory: {output}")
    if output.resolve() == weights.resolve():
        raise ValueError("output must differ from the source model")
    if (
        not isinstance(input_size, numbers.Integral)
        or isinstance(input_size, bool)
        or input_size < MODEL_STRIDE
        or input_size % MODEL_STRIDE
    ):
        raise ValueError(
            f"input_size must be an integer >= {MODEL_STRIDE} and a multiple of {MODEL_STRIDE}, got {input_size!r}"
        )
    if opset is not None and (
        not isinstance(opset, numbers.Integral) or isinstance(opset, bool) or opset < MIN_OPSET
    ):
        raise ValueError(f"opset must be None (automatic) or an integer >= {MIN_OPSET}, got {opset!r}")
    for name, value in (("simplify", simplify), ("dynamic", dynamic)):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be True or False, got {value!r}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists; use overwrite=True (--overwrite) to replace it")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_onnx(
    weights: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    input_size: int = 640,
    opset: int | None = None,
    simplify: bool = True,
    dynamic: bool = False,
    overwrite: bool = False,
) -> ExportResult:
    """Export a YOLO detection model to ONNX with Ultralytics, on the CPU.

    Args:
        weights: Existing ``.pt`` detection model (never downloaded here).
        output: Where to write the ``.onnx`` file; parent folders are created.
        input_size: Square input size the graph is built for (multiple of 32).
        opset: ONNX opset version, or ``None`` to let Ultralytics choose the
            best one for the installed PyTorch / ONNX.
        simplify: Simplify the graph with onnxslim (Ultralytics' default).
        dynamic: Allow variable batch size and image size in the exported graph.
        overwrite: Replace ``output`` if it exists (only once the export succeeded).

    Raises:
        ExportError: missing or invalid source model, failed export, or no/empty output.
        FileExistsError: ``output`` exists and ``overwrite`` is False.
        ValueError: invalid argument.
        MissingDependencyError: a package needed for export is not installed.
    """
    weights, output = Path(weights), Path(output)
    _validate_arguments(weights, output, input_size, opset, simplify, dynamic, overwrite)
    check_export_dependencies(simplify)
    from ultralytics import YOLO

    with tempfile.TemporaryDirectory(prefix="edgevision-export-") as tmp:
        source_copy = Path(tmp) / weights.name
        shutil.copy2(weights, source_copy)
        try:
            model = YOLO(str(source_copy))
        except Exception as exc:
            raise ExportError(f"Could not load {weights}: {type(exc).__name__}: {exc}") from exc
        task = getattr(model, "task", None)
        if task != "detect":
            raise ExportError(f"{weights} is a {task!r} model; only detection models are supported")

        logger.info("Exporting %s to ONNX (input %d, opset %s)", weights, input_size, opset or "auto")
        try:
            produced = model.export(
                format="onnx",
                imgsz=int(input_size),
                opset=None if opset is None else int(opset),
                simplify=simplify,
                dynamic=dynamic,
                device="cpu",
                # Left at Ultralytics' defaults on purpose: FP32 (no half/quantize), and NMS
                # outside the graph (nms unset; nms=False would select a different head).
            )
        except Exception as exc:
            raise ExportError(f"Ultralytics ONNX export failed: {type(exc).__name__}: {exc}") from exc

        expected = source_copy.with_suffix(".onnx")
        produced_path = expected if expected.exists() else Path(str(produced or ""))
        if not produced_path.is_file():
            raise ExportError(f"The export reported success but no ONNX file was created (expected {expected}).")
        size = produced_path.stat().st_size
        if size == 0:
            raise ExportError(f"The export produced an empty file ({produced_path}).")

        output.parent.mkdir(parents=True, exist_ok=True)
        partial = partial_path_for(output)
        try:
            shutil.copyfile(produced_path, partial)
            os.replace(partial, output)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise

    return ExportResult(
        path=output,
        size_bytes=output.stat().st_size,
        source=weights,
        input_size=int(input_size),
        opset_requested=None if opset is None else int(opset),
        simplify=simplify,
        dynamic=dynamic,
    )
