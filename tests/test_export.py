"""Tests for edgevision.models.export.

Unit tests replace PyTorch and Ultralytics with fakes and control which
packages appear installed, so they need no model, no network, no GPU and no
ONNX packages. The fake ``YOLO.export`` writes a small placeholder file where
Ultralytics would write the real one.

One opt-in test (marker ``model``) runs the real Ultralytics export.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import edgevision.models.export as export_module
from edgevision.models.export import ExportError, ExportResult, export_onnx
from edgevision.optional import MissingDependencyError

REPO = Path(__file__).resolve().parents[1]
INSTALLED = {"onnx": "1.23.2", "onnxslim": "0.1.98", "onnxruntime": "1.31.0"}


@dataclass
class FakeUltralytics:
    task: str = "detect"
    mode: str = "ok"  # ok | raise | no_file | empty
    load_error: Exception | None = None
    loaded: list[Path] = field(default_factory=list)
    export_calls: list[dict[str, Any]] = field(default_factory=list)
    versions: dict[str, str] = field(default_factory=lambda: dict(INSTALLED))


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeUltralytics:
    ctl = FakeUltralytics()

    class FakeYOLO:
        def __init__(self, path: str) -> None:
            ctl.loaded.append(Path(path))
            if ctl.load_error is not None:
                raise ctl.load_error
            self.path = Path(path)
            self.task = ctl.task

        def export(self, **kwargs: Any) -> str:
            ctl.export_calls.append(kwargs)
            target = self.path.with_suffix(".onnx")
            if ctl.mode == "raise":
                raise RuntimeError("ONNX export failure: Unsupported operator")
            if ctl.mode == "ok":
                target.write_bytes(b"fake onnx bytes")
            elif ctl.mode == "empty":
                target.write_bytes(b"")
            return str(target)  # "no_file": reports a path that was never written

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="2.14.1+cpu"))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    monkeypatch.setattr(export_module, "_installed_version", lambda dist: ctl.versions.get(dist))
    return ctl


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    path = tmp_path / "models" / "yolov8n.pt"
    path.parent.mkdir()
    path.write_bytes(b"fake weights")
    return path


# --- success ---------------------------------------------------------------------------


def test_successful_export(fake: FakeUltralytics, weights: Path, tmp_path: Path) -> None:
    output = tmp_path / "exports" / "nested" / "model.onnx"  # parent folders do not exist yet

    result = export_onnx(weights, output)

    assert result == ExportResult(
        path=output, size_bytes=len(b"fake onnx bytes"), source=weights,
        input_size=640, opset_requested=None, simplify=True, dynamic=False,
    )
    assert output.read_bytes() == b"fake onnx bytes"
    assert fake.export_calls == [
        {"format": "onnx", "imgsz": 640, "opset": None, "simplify": True, "dynamic": False, "device": "cpu"}
    ]
    assert not (output.parent / "model.partial.onnx").exists()


def test_export_never_writes_next_to_the_weights(fake: FakeUltralytics, weights: Path, tmp_path: Path) -> None:
    neighbour = weights.with_suffix(".onnx")
    neighbour.write_bytes(b"someone else's onnx")  # Ultralytics would overwrite this
    export_onnx(weights, tmp_path / "out.onnx")
    assert fake.loaded[0] != weights and fake.loaded[0].name == weights.name  # exported from a temporary copy
    assert not fake.loaded[0].parent.exists()  # and the temporary folder was removed
    assert neighbour.read_bytes() == b"someone else's onnx"
    assert weights.read_bytes() == b"fake weights"


def test_options_are_passed_to_ultralytics(fake: FakeUltralytics, weights: Path, tmp_path: Path) -> None:
    result = export_onnx(weights, tmp_path / "o.onnx", input_size=320, opset=17, simplify=False, dynamic=True)
    call = fake.export_calls[0]
    assert (call["imgsz"], call["opset"], call["simplify"], call["dynamic"]) == (320, 17, False, True)
    assert (result.input_size, result.opset_requested, result.simplify, result.dynamic) == (320, 17, False, True)


# --- invalid source / output / options ---------------------------------------------------


def test_missing_source_model(fake: FakeUltralytics, tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="Model file not found.*never downloads"):
        export_onnx(tmp_path / "missing.pt", tmp_path / "out.onnx")
    assert fake.loaded == []


def test_source_that_is_a_directory_or_not_pt(fake: FakeUltralytics, tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="not a file"):
        export_onnx(tmp_path, tmp_path / "out.onnx")
    other = tmp_path / "model.onnx"
    other.write_bytes(b"x")
    with pytest.raises(ExportError, match=r"Expected a PyTorch \.pt model"):
        export_onnx(other, tmp_path / "out.onnx")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"input_size": 641}, "multiple of 32"),
        ({"input_size": 0}, "input_size must be"),
        ({"input_size": True}, "input_size must be"),
        ({"input_size": 640.0}, "input_size must be"),
        ({"opset": 5}, "opset must be None"),
        ({"opset": "18"}, "opset must be None"),
        ({"simplify": "yes"}, "simplify must be True or False"),
        ({"dynamic": 1}, "dynamic must be True or False"),
    ],
)
def test_invalid_options(fake: FakeUltralytics, weights: Path, tmp_path: Path, kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        export_onnx(weights, tmp_path / "out.onnx", **kwargs)
    assert fake.export_calls == []


def test_invalid_output_paths(fake: FakeUltralytics, weights: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"must end in \.onnx"):
        export_onnx(weights, tmp_path / "out.onnx.txt")
    folder = tmp_path / "folder.onnx"
    folder.mkdir()
    with pytest.raises(ValueError, match="directory"):
        export_onnx(weights, folder)


# --- overwrite protection -----------------------------------------------------------------


def test_existing_output_is_not_overwritten(fake: FakeUltralytics, weights: Path, tmp_path: Path) -> None:
    output = tmp_path / "out.onnx"
    output.write_bytes(b"previous export")
    with pytest.raises(FileExistsError, match="already exists"):
        export_onnx(weights, output)
    assert output.read_bytes() == b"previous export"
    assert fake.export_calls == []  # refused before doing any work


def test_overwrite_replaces_existing_output(fake: FakeUltralytics, weights: Path, tmp_path: Path) -> None:
    output = tmp_path / "out.onnx"
    output.write_bytes(b"previous export")
    export_onnx(weights, output, overwrite=True)
    assert output.read_bytes() == b"fake onnx bytes"


def test_failed_export_keeps_the_previous_output(fake: FakeUltralytics, weights: Path, tmp_path: Path) -> None:
    output = tmp_path / "out.onnx"
    output.write_bytes(b"previous export")
    fake.mode = "raise"
    with pytest.raises(ExportError):
        export_onnx(weights, output, overwrite=True)
    assert output.read_bytes() == b"previous export"


# --- export failures ---------------------------------------------------------------------------


def test_export_failure_is_reported(fake: FakeUltralytics, weights: Path, tmp_path: Path) -> None:
    fake.mode = "raise"
    output = tmp_path / "out.onnx"
    with pytest.raises(ExportError, match="Ultralytics ONNX export failed: RuntimeError: .*Unsupported operator") as info:
        export_onnx(weights, output)
    assert isinstance(info.value.__cause__, RuntimeError)
    assert not output.exists() and not (tmp_path / "out.partial.onnx").exists()


@pytest.mark.parametrize(("mode", "message"), [("no_file", "no ONNX file was created"), ("empty", "empty file")])
def test_missing_or_empty_artifact(fake: FakeUltralytics, weights: Path, tmp_path: Path, mode: str, message: str) -> None:
    fake.mode = mode
    output = tmp_path / "out.onnx"
    with pytest.raises(ExportError, match=message):
        export_onnx(weights, output)
    assert not output.exists()


def test_model_that_cannot_be_loaded(fake: FakeUltralytics, weights: Path, tmp_path: Path) -> None:
    fake.load_error = RuntimeError("invalid load key, '\\x00'")
    with pytest.raises(ExportError, match="Could not load .*invalid load key"):
        export_onnx(weights, tmp_path / "out.onnx")


def test_non_detection_model_is_rejected(fake: FakeUltralytics, weights: Path, tmp_path: Path) -> None:
    fake.task = "segment"
    with pytest.raises(ExportError, match="'segment' model"):
        export_onnx(weights, tmp_path / "out.onnx")
    assert fake.export_calls == []


# --- dependencies ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("versions", "simplify", "message"),
    [
        ({"onnxslim": "0.1.98", "onnxruntime": "1.31.0"}, True, "onnx is required"),
        ({"onnx": "2.0.0", "onnxslim": "0.1.98", "onnxruntime": "1.31.0"}, True, "onnx 2.0.0 is installed, but"),
        ({"onnx": "1.23.2", "onnxruntime": "1.31.0"}, True, "onnxslim is required"),
        ({"onnx": "1.23.2", "onnxslim": "0.1.50", "onnxruntime": "1.31.0"}, True, "onnxslim 0.1.50 is installed"),
        ({"onnx": "1.23.2", "onnxslim": "0.1.98"}, True, "ONNX Runtime package is required.*--no-simplify"),
    ],
)
def test_missing_export_dependencies(
    fake: FakeUltralytics, weights: Path, tmp_path: Path, versions: dict, simplify: bool, message: str
) -> None:
    fake.versions = versions
    with pytest.raises(MissingDependencyError, match=message) as info:
        export_onnx(weights, tmp_path / "out.onnx", simplify=simplify)
    assert 'pip install -e ".[export]"' in str(info.value)
    assert fake.loaded == []  # Ultralytics never got the chance to auto-install anything


def test_without_simplify_only_onnx_is_needed(fake: FakeUltralytics, weights: Path, tmp_path: Path) -> None:
    fake.versions = {"onnx": "1.23.2"}
    export_onnx(weights, tmp_path / "out.onnx", simplify=False)
    assert fake.export_calls[0]["simplify"] is False


def test_any_onnxruntime_variant_is_accepted(fake: FakeUltralytics, weights: Path, tmp_path: Path) -> None:
    fake.versions = {"onnx": "1.23.2", "onnxslim": "0.1.98", "onnxruntime-gpu": "1.31.0"}
    export_onnx(weights, tmp_path / "out.onnx")
    assert fake.export_calls


@pytest.mark.parametrize(("module", "name"), [("torch", "PyTorch"), ("ultralytics", "Ultralytics")])
def test_missing_detector_dependencies(
    fake: FakeUltralytics, weights: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module: str, name: str
) -> None:
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(MissingDependencyError, match=rf'{name} is required.*\.\[detector\]'):
        export_onnx(weights, tmp_path / "out.onnx")


def test_importing_export_modules_needs_no_optional_packages() -> None:
    code = (
        "import sys\n"
        "for name in ('torch', 'ultralytics', 'onnx', 'onnxslim', 'onnxruntime', 'cv2', 'numpy'):\n"
        "    sys.modules[name] = None\n"
        "import edgevision.models.export, edgevision.models.onnx_validation\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


# --- real export (opt-in) -------------------------------------------------------------------------

RUN_MODEL_TESTS = os.environ.get("EDGEVISION_RUN_MODEL_TESTS") == "1"
WEIGHTS = Path(os.environ.get("EDGEVISION_MODEL_WEIGHTS", REPO / "models" / "yolov8n.pt"))


@pytest.mark.model
@pytest.mark.skipif(
    not RUN_MODEL_TESTS,
    reason="real-model test is opt-in: set EDGEVISION_RUN_MODEL_TESTS=1 (needs the detector and export extras and models/yolov8n.pt)",
)
def test_real_export_and_validation(tmp_path: Path) -> None:
    if not WEIGHTS.is_file():
        pytest.skip(f"weights not found at {WEIGHTS}; run scripts/detect_image.py once to download them")
    pytest.importorskip("ultralytics")
    pytest.importorskip("onnx", reason='export extra not installed (pip install -e ".[export]")')
    from edgevision.models.onnx_validation import validate_onnx, yolo_layout_problems

    result = export_onnx(WEIGHTS, tmp_path / "yolov8n_320.onnx", input_size=320)
    info = validate_onnx(result.path)

    assert result.size_bytes > 1_000_000
    assert [t.shape for t in info.inputs] == [(1, 3, 320, 320)]
    assert [t.shape for t in info.outputs] == [(1, 84, 2100)]  # 4 box values + 80 classes, 2100 candidates
    assert info.metadata["task"] == "detect" and len(info.class_names) == 80
    assert yolo_layout_problems(info, 320) == []
    assert not list(WEIGHTS.parent.glob("*.partial.onnx"))
