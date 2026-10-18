"""Tests for scripts/export_onnx.py and scripts/inspect_onnx.py.

``--help`` and argument errors run as real subprocesses. Other tests import the
script and replace ``export_onnx`` / ``validate_onnx`` with fakes, so they
check the command's own logic (config defaults, options, messages, exit
codes) without exporting anything.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from edgevision.models.export import ExportError, ExportResult
from edgevision.models.onnx_validation import OnnxModelInfo, OnnxValidationError, TensorInfo
from edgevision.optional import MissingDependencyError

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "default.yaml"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"{name}_script", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(REPO / "scripts" / f"{name}.py"), *args], capture_output=True, text=True, check=False)


def good_info(path: Path, size: int = 640) -> OnnxModelInfo:
    return OnnxModelInfo(
        path=path, size_bytes=12_851_106, ir_version=8, opset=18, producer="pytorch 2.14.1",
        inputs=(TensorInfo("images", "FLOAT", (1, 3, size, size)),),
        outputs=(TensorInfo("output0", "FLOAT", (1, 84, 8400)),),
        metadata={"task": "detect"},
    )


@pytest.fixture
def export_script(tmp_path: Path):
    """export_onnx.py with fake export/validation that record their arguments."""
    script = load("export_onnx")
    calls: dict[str, Any] = {"export": [], "validate": []}

    def fake_export(source, output, **kwargs):
        calls["export"].append((Path(source), Path(output), kwargs))
        return ExportResult(Path(output), 12_851_106, Path(source), kwargs["input_size"], kwargs["opset"], kwargs["simplify"], kwargs["dynamic"])

    def fake_validate(path):
        calls["validate"].append(Path(path))
        return good_info(Path(path), size=calls["export"][-1][2]["input_size"])

    script.export_onnx = fake_export
    script.validate_onnx = fake_validate
    script.calls = calls
    return script


# --- export_onnx.py --------------------------------------------------------------------------


def test_help_does_not_export(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "export_onnx.py"), "--help"],
        capture_output=True, text=True, check=False, cwd=tmp_path,
    )
    assert result.returncode == 0
    for option in ("--model", "--output", "--imgsz", "--opset", "--dynamic", "--no-simplify", "--overwrite", "--validate"):
        assert option in result.stdout
    assert list(tmp_path.iterdir()) == []  # nothing was created


@pytest.mark.parametrize("args", [["--imgsz", "0"], ["--opset", "x"], ["--imgsz", "abc"]])
def test_invalid_arguments(args: list[str]) -> None:
    result = run("export_onnx", *args)
    assert result.returncode == 2 and "usage:" in result.stderr


def test_defaults_come_from_the_config(export_script, capsys: pytest.CaptureFixture[str]) -> None:
    assert export_script.main(["--config", str(CONFIG)]) == 0
    source, output, kwargs = export_script.calls["export"][0]
    assert (source, output) == (Path("models/yolov8n.pt"), Path("models/yolov8n.onnx"))
    assert kwargs == {"input_size": 640, "opset": None, "simplify": True, "dynamic": False, "overwrite": False}
    assert export_script.calls["validate"] == [Path("models/yolov8n.onnx")]
    out = capsys.readouterr().out
    # The path is printed in the platform's own style (models\yolov8n.onnx on Windows).
    assert f"Exported: {Path('models/yolov8n.onnx')} (12.3 MiB, 12,851,106 bytes)" in out
    assert "ONNX checker passed" in out and "does not prove" in out


def test_options_override_the_config(export_script, tmp_path: Path) -> None:
    args = ["--model", "m.pt", "--output", str(tmp_path / "o.onnx"), "--imgsz", "320", "--opset", "17",
            "--dynamic", "--no-simplify", "--overwrite", "--config", str(CONFIG)]
    assert export_script.main(args) == 0
    source, output, kwargs = export_script.calls["export"][0]
    assert (source, output) == (Path("m.pt"), tmp_path / "o.onnx")
    assert kwargs == {"input_size": 320, "opset": 17, "simplify": False, "dynamic": True, "overwrite": True}


def test_config_export_section_is_used(export_script, tmp_path: Path) -> None:
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data["export"].update(opset=12, dynamic_shapes=True, validate=False)
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert export_script.main(["--config", str(config)]) == 0
    assert export_script.calls["export"][0][2]["opset"] == 12
    assert export_script.calls["export"][0][2]["dynamic"] is True
    assert export_script.calls["validate"] == []  # export.validate: false


def test_no_validate_skips_validation(export_script, capsys: pytest.CaptureFixture[str]) -> None:
    assert export_script.main(["--no-validate", "--config", str(CONFIG)]) == 0
    assert export_script.calls["validate"] == []
    assert "Validation skipped" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ExportError("Model file not found: models/missing.pt"), "Model file not found"),
        (FileExistsError("models/yolov8n.onnx already exists"), "already exists"),
        (MissingDependencyError('onnx is required ... ".[export]"'), ".[export]"),
    ],
)
def test_export_failures_exit_with_status_1(export_script, capsys, error: Exception, expected: str) -> None:
    def failing(*args, **kwargs):
        raise error

    export_script.export_onnx = failing
    assert export_script.main(["--config", str(CONFIG)]) == 1
    assert expected in capsys.readouterr().err


def test_invalid_option_values_exit_with_status_2(export_script, capsys) -> None:
    def failing(*args, **kwargs):
        raise ValueError("input_size must be an integer >= 32 and a multiple of 32, got 641")

    export_script.export_onnx = failing
    with pytest.raises(SystemExit) as info:
        export_script.main(["--imgsz", "641", "--config", str(CONFIG)])
    assert info.value.code == 2
    assert "multiple of 32" in capsys.readouterr().err


def test_validation_failure_exits_with_status_1(export_script, capsys) -> None:
    def invalid(path):
        raise OnnxValidationError("models/yolov8n.onnx is not a valid ONNX model: ...")

    export_script.validate_onnx = invalid
    assert export_script.main(["--config", str(CONFIG)]) == 1
    assert "exported file failed validation" in capsys.readouterr().err


def test_unexpected_layout_exits_with_status_1(export_script, capsys) -> None:
    export_script.validate_onnx = lambda path: good_info(Path(path), size=320)  # but 640 was requested
    assert export_script.main(["--config", str(CONFIG)]) == 1
    assert "input height is 320, expected 640" in capsys.readouterr().err


# --- inspect_onnx.py ---------------------------------------------------------------------------


def test_inspect_help() -> None:
    result = run("inspect_onnx", "--help")
    assert result.returncode == 0 and "--imgsz" in result.stdout


def test_inspect_reports_a_valid_model(capsys) -> None:
    script = load("inspect_onnx")
    script.validate_onnx = lambda path: good_info(Path(path))
    assert script.main(["models/yolov8n.onnx", "--imgsz", "640"]) == 0
    out = " ".join(capsys.readouterr().out.split())
    assert "PASSED" in out and "Opset 18" in out
    assert "Input images: FLOAT [1, 3, 640, 640]" in out and "Output output0: FLOAT [1, 84, 8400]" in out
    assert "structural checks only" in out


def test_inspect_flags_wrong_size(capsys) -> None:
    script = load("inspect_onnx")
    script.validate_onnx = lambda path: good_info(Path(path))
    assert script.main(["models/yolov8n.onnx", "--imgsz", "320"]) == 1
    assert "NOT as expected" in capsys.readouterr().out


def test_inspect_invalid_or_missing_file(tmp_path: Path) -> None:
    result = run("inspect_onnx", str(tmp_path / "missing.onnx"))
    assert result.returncode == 1 and "ONNX file not found" in result.stderr
    bad = tmp_path / "empty.onnx"
    bad.write_bytes(b"")
    result = run("inspect_onnx", str(bad))
    assert result.returncode == 1 and "empty" in result.stderr
