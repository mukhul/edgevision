"""Tests for edgevision.models.onnx_validation.

* Layout checks work on hand-made ``OnnxModelInfo`` values (pure Python).
* ``validate_onnx`` is tested against small *real* ONNX files built with
  ``onnx.helper`` when the ``onnx`` package is installed (export extra), and
  against a fake ``onnx`` module otherwise, so the logic is always covered.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from edgevision.models.onnx_validation import (
    OnnxModelInfo,
    OnnxValidationError,
    TensorInfo,
    validate_onnx,
    yolo_layout_problems,
)
from edgevision.optional import MissingDependencyError

NAMES = "{" + ", ".join(f"{i}: 'class{i}'" for i in range(80)) + "}"


def info(inputs=None, outputs=None, metadata=None) -> OnnxModelInfo:
    return OnnxModelInfo(
        path=Path("m.onnx"), size_bytes=10, ir_version=8, opset=18, producer="pytorch 2.14.1",
        inputs=tuple(inputs or [TensorInfo("images", "FLOAT", (1, 3, 640, 640))]),
        outputs=tuple(outputs or [TensorInfo("output0", "FLOAT", (1, 84, 8400))]),
        metadata={"task": "detect", "names": NAMES} if metadata is None else metadata,
    )


# --- missing / empty files (no onnx needed) ------------------------------------------------


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(OnnxValidationError, match="ONNX file not found"):
        validate_onnx(tmp_path / "missing.onnx")


def test_directory(tmp_path: Path) -> None:
    with pytest.raises(OnnxValidationError, match="not a file"):
        validate_onnx(tmp_path)


def test_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.onnx"
    path.write_bytes(b"")
    with pytest.raises(OnnxValidationError, match="empty"):
        validate_onnx(path)


def test_missing_onnx_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "m.onnx"
    path.write_bytes(b"x")
    monkeypatch.setitem(sys.modules, "onnx", None)
    with pytest.raises(MissingDependencyError, match=r'onnx is required to validate ONNX models.*\.\[export\]'):
        validate_onnx(path)


# --- YOLO layout (pure Python) -----------------------------------------------------------------


def test_expected_layout_has_no_problems() -> None:
    assert yolo_layout_problems(info(), input_size=640) == []


def test_dynamic_dimensions_are_accepted() -> None:
    dynamic = info(
        inputs=[TensorInfo("images", "FLOAT", ("batch", 3, "height", "width"))],
        outputs=[TensorInfo("output0", "FLOAT", ("batch", 84, "anchors"))],
    )
    assert yolo_layout_problems(dynamic, input_size=320) == []


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"inputs": [TensorInfo("images", "FLOAT", (1, 3, 320, 320))]}, "input height is 320, expected 640"),
        ({"inputs": [TensorInfo("images", "FLOAT", (1, 1, 640, 640))]}, "1 channels, expected 3"),
        ({"inputs": [TensorInfo("images", "FLOAT16", (1, 3, 640, 640))]}, "is FLOAT16, expected FLOAT"),
        ({"inputs": [TensorInfo("images", "FLOAT", (3, 640, 640))]}, "expected (batch, 3, height, width)"),
        ({"outputs": [TensorInfo("output0", "FLOAT", (1, 85, 8400))]}, "85 values per candidate, expected 4 box values + 80 classes"),
        ({"outputs": [TensorInfo("output0", "FLOAT", (1, 8400))]}, "expected (batch, 4 + classes, candidates)"),
        ({"outputs": [TensorInfo("a", "FLOAT", (1, 84, 10)), TensorInfo("b", "FLOAT", (1, 32, 160, 160))]}, "expected 1 output, found 2"),
        ({"metadata": {"task": "segment"}}, "task 'segment'"),
    ],
)
def test_layout_problems_are_reported(changes: dict, message: str) -> None:
    problems = yolo_layout_problems(info(**changes), input_size=640)
    assert any(message in p for p in problems), problems


def test_class_names_are_read_safely() -> None:
    assert len(info().class_names) == 80
    assert info(metadata={}).class_names is None
    # The metadata is parsed as a literal, never executed.
    assert info(metadata={"names": "__import__('os').system('echo hacked')"}).class_names is None
    assert info(metadata={"names": "['a', 'b']"}).class_names is None


def test_shape_text() -> None:
    assert TensorInfo("x", "FLOAT", ("batch", 3, None, 640)).shape_text == "[batch, 3, ?, 640]"


# --- validate_onnx with a fake onnx module (always runs) ----------------------------------------


class FakeDim:
    def __init__(self, value: int | None = None, param: str = "") -> None:
        self.dim_value, self.dim_param = value or 0, param
        self._has_value = value is not None

    def HasField(self, name: str) -> bool:  # mirrors protobuf's HasField
        return self._has_value if name == "dim_value" else bool(self.dim_param)


def fake_value(name: str, elem_type: int, dims: list[FakeDim]) -> Any:
    return SimpleNamespace(name=name, type=SimpleNamespace(tensor_type=SimpleNamespace(elem_type=elem_type, shape=SimpleNamespace(dim=dims))))


def install_fake_onnx(monkeypatch: pytest.MonkeyPatch, check_error: Exception | None = None) -> None:
    def check_model(path: str, full_check: bool = False) -> None:
        assert full_check is True
        if check_error is not None:
            raise check_error

    model = SimpleNamespace(
        ir_version=8, producer_name="pytorch", producer_version="2.14.1",
        opset_import=[SimpleNamespace(domain="", version=18)],
        metadata_props=[SimpleNamespace(key="task", value="detect")],
        graph=SimpleNamespace(
            initializer=[SimpleNamespace(name="weight")],
            input=[fake_value("images", 1, [FakeDim(param="batch"), FakeDim(3), FakeDim(640), FakeDim(640)]),
                   fake_value("weight", 1, [FakeDim(16)])],  # an initializer listed as input (old IR style)
            output=[fake_value("output0", 1, [FakeDim(param="batch"), FakeDim(84), FakeDim()])],
        ),
    )
    names = {1: "FLOAT"}

    def name(value: int) -> str:
        if value not in names:
            raise ValueError(value)
        return names[value]

    monkeypatch.setitem(sys.modules, "onnx", SimpleNamespace(
        checker=SimpleNamespace(check_model=check_model),
        load=lambda path, load_external_data=True: model,
        TensorProto=SimpleNamespace(DataType=SimpleNamespace(Name=name)),
    ))


def test_mocked_valid_model_is_described(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "m.onnx"
    path.write_bytes(b"pretend onnx")
    install_fake_onnx(monkeypatch)

    result = validate_onnx(path)

    assert (result.opset, result.ir_version, result.producer, result.size_bytes) == (18, 8, "pytorch 2.14.1", 12)
    assert result.inputs == (TensorInfo("images", "FLOAT", ("batch", 3, 640, 640)),)  # initializer excluded
    assert result.outputs == (TensorInfo("output0", "FLOAT", ("batch", 84, None)),)
    assert result.metadata == {"task": "detect"}


def test_mocked_checker_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "m.onnx"
    path.write_bytes(b"pretend onnx")
    install_fake_onnx(monkeypatch, check_error=RuntimeError("Node (Conv_0) has input size 0"))
    with pytest.raises(OnnxValidationError, match="not a valid ONNX model: RuntimeError: Node") as info_:
        validate_onnx(path)
    assert isinstance(info_.value.__cause__, RuntimeError)


# --- validate_onnx with real ONNX files (needs the export extra) ---------------------------------


@pytest.fixture
def onnx() -> Any:
    return pytest.importorskip("onnx", reason='onnx not installed (pip install -e ".[export]")')


def save_model(onnx: Any, path: Path, *, nodes: list, inputs: list, outputs: list, metadata: dict | None = None) -> Path:
    from onnx import helper

    graph = helper.make_graph(nodes, "test", inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="edgevision-test")
    model.ir_version = 8
    for key, value in (metadata or {}).items():
        prop = model.metadata_props.add()
        prop.key, prop.value = key, value
    onnx.save(model, str(path))
    return path


def test_real_valid_model(onnx: Any, tmp_path: Path) -> None:
    from onnx import TensorProto, helper

    path = save_model(
        onnx, tmp_path / "identity.onnx",
        nodes=[helper.make_node("Identity", ["images"], ["output0"])],
        inputs=[helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 3, 64, 64])],
        outputs=[helper.make_tensor_value_info("output0", TensorProto.FLOAT, ["batch", 3, 64, 64])],
        metadata={"task": "detect", "imgsz": "[64, 64]"},
    )
    result = validate_onnx(path)
    assert result.opset == 17 and result.ir_version == 8 and result.producer == "edgevision-test"
    assert result.inputs == (TensorInfo("images", "FLOAT", ("batch", 3, 64, 64)),)
    assert result.outputs == (TensorInfo("output0", "FLOAT", ("batch", 3, 64, 64)),)
    assert result.metadata == {"task": "detect", "imgsz": "[64, 64]"}
    # A valid ONNX file need not be a YOLO model: the layout check says so.
    assert yolo_layout_problems(result, 64)


def test_real_model_rejected_by_checker(onnx: Any, tmp_path: Path) -> None:
    from onnx import TensorProto, helper

    path = save_model(
        onnx, tmp_path / "broken.onnx",
        nodes=[helper.make_node("Identity", ["does_not_exist"], ["output0"])],  # undefined input
        inputs=[helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3])],
        outputs=[helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 3])],
    )
    with pytest.raises(OnnxValidationError, match="not a valid ONNX model"):
        validate_onnx(path)


@pytest.mark.parametrize("content", [b"definitely not protobuf", b"\x08\x08\x12\xff\xff"])
def test_real_malformed_files_are_rejected(onnx: Any, tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "bad.onnx"
    path.write_bytes(content)
    with pytest.raises(OnnxValidationError, match="not a valid ONNX model"):
        validate_onnx(path)
