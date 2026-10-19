"""Every command must work in a base installation (PyYAML only) as far as it can.

Each command runs in a subprocess in which the optional packages are blocked
(``sys.modules[name] = None`` makes ``import name`` fail exactly as if it were
not installed). ``--help`` must always work, and a command that needs a
missing package must stop with exit status 1 and an install hint, never with
a traceback.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
OPTIONAL = ("numpy", "cv2", "torch", "ultralytics", "onnx", "onnxruntime", "onnxslim", "lap", "matplotlib")
SCRIPTS = sorted(p.stem for p in (REPO / "scripts").glob("*.py"))

BOOTSTRAP = (
    "import runpy, sys\n"
    "for name in sys.argv[1].split(','):\n"
    "    sys.modules[name] = None\n"
    "script = sys.argv[2]\n"
    "sys.argv = [script] + sys.argv[3:]\n"
    "runpy.run_path(script, run_name='__main__')\n"
)


def run_without_extras(script: str, *args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", BOOTSTRAP, ",".join(OPTIONAL), str(REPO / "scripts" / f"{script}.py"), *args],
        capture_output=True, text=True, check=False, cwd=cwd,
    )


def test_every_script_is_covered() -> None:
    assert "compare_detections" in SCRIPTS and "evaluate" in SCRIPTS and len(SCRIPTS) >= 10


@pytest.mark.parametrize("script", SCRIPTS)
def test_help_works_without_optional_packages(script: str, tmp_path: Path) -> None:
    result = run_without_extras(script, "--help", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout and "Traceback" not in result.stderr
    assert list(tmp_path.iterdir()) == []  # --help writes nothing


def test_compare_detections_without_numpy_gives_install_hint(tmp_path: Path) -> None:
    doc = {"schema": "edgevision.video_detections", "schema_version": 1, "frames": []}
    for name in ("a.json", "b.json"):
        (tmp_path / name).write_text(json.dumps(doc), encoding="utf-8")
    result = run_without_extras("compare_detections", "a.json", "b.json", cwd=tmp_path)
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "NumPy is required to compute detection metrics" in result.stderr
    assert 'pip install -e ".[vision]"' in result.stderr


def test_evaluate_without_numpy_gives_install_hint(tmp_path: Path) -> None:
    result = run_without_extras("evaluate", "--annotations", "a.json", "--images", "images", cwd=tmp_path)
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "NumPy is required to compute detection metrics" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_argument_errors_still_exit_2_without_numpy(tmp_path: Path) -> None:
    result = run_without_extras("compare_detections", "a.json", "b.json", "--iou", "0", cwd=tmp_path)
    assert result.returncode == 2 and "--iou must be greater than 0" in result.stderr
    result = run_without_extras("evaluate", "--images", "images", cwd=tmp_path)
    assert result.returncode == 2 and "--annotations" in result.stderr
