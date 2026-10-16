"""Tests for edgevision.device.

PyTorch, ONNX Runtime and nvidia-smi are replaced with small fakes, so these
tests pass on any machine: no GPU, no ML packages, and no network needed.

Mocking techniques used:

* ``sys.modules["torch"] = None`` makes ``import torch`` fail exactly as if
  torch were not installed.
* Putting a fake object in ``sys.modules`` makes ``import torch`` return it.
* ``monkeypatch`` swaps a function for the duration of one test only.
"""

from __future__ import annotations

import importlib.abc
import json
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from edgevision import device


# --- fakes -------------------------------------------------------------------


def flat(text: str) -> str:
    """Collapse runs of spaces so assertions do not depend on column alignment."""
    return " ".join(text.split())


def fake_torch(cuda_build: str | None, cuda_available: bool, gpus: tuple[str, ...] = ()) -> Any:
    return SimpleNamespace(
        __version__="2.4.0",
        version=SimpleNamespace(cuda=cuda_build),
        cuda=SimpleNamespace(
            is_available=lambda: cuda_available,
            device_count=lambda: len(gpus),
            get_device_name=lambda i: gpus[i],
        ),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 90100)),
    )


def fake_onnxruntime(providers: list[str]) -> Any:
    return SimpleNamespace(__version__="1.19.2", get_available_providers=lambda: providers)


def fake_nvidia_smi(monkeypatch: pytest.MonkeyPatch, stdout: str, returncode: int = 0) -> None:
    monkeypatch.setattr(device.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr="driver error")

    monkeypatch.setattr(device.subprocess, "run", run)


@pytest.fixture
def bare_machine(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """A CPU-only machine with none of the optional packages installed."""
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    monkeypatch.setattr(device, "_installed_version", lambda dist: None)
    monkeypatch.setattr(device.shutil, "which", lambda name: None)
    return monkeypatch


# --- no optional packages, no GPU --------------------------------------------


def test_report_without_optional_packages(bare_machine: pytest.MonkeyPatch) -> None:
    env = device.collect_environment()

    assert env["torch"] == {"installed": False}
    assert env["onnxruntime"] == {"installed": False}
    assert env["nvidia_gpus"]["gpus"] == []
    assert set(env["missing_optional_packages"]) == set(device.OPTIONAL_PACKAGES)

    text = flat(device.format_report(env))
    assert "NVIDIA GPU none detected (nvidia-smi not found)" in text
    assert "PyTorch not installed" in text
    assert "ONNX Runtime not installed" in text
    assert "it cannot show that GPU inference actually works" in text


def test_basic_system_information_is_reported(bare_machine: pytest.MonkeyPatch) -> None:
    env = device.collect_environment()
    assert env["python"]["version"].count(".") == 2
    assert env["os"]["system"]
    assert "logical_cores" in env["cpu"]


def test_cli_text_and_json(bare_machine: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    assert device.main([]) == 0
    assert "environment report" in capsys.readouterr().out

    assert device.main(["--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["torch"]["installed"] is False
    assert "warnings" in data


# --- PyTorch -----------------------------------------------------------------


def test_torch_with_usable_cuda(bare_machine: pytest.MonkeyPatch) -> None:
    bare_machine.setitem(sys.modules, "torch", fake_torch("12.1", True, ("NVIDIA Test GPU",)))

    info = device.probe_torch()

    assert info["version"] == "2.4.0"
    assert info["cuda_build"] == "12.1"
    assert info["cuda_available"] is True
    assert info["cuda_devices"] == ["NVIDIA Test GPU"]


def test_cpu_only_torch_on_a_gpu_machine_is_flagged(bare_machine: pytest.MonkeyPatch) -> None:
    bare_machine.setitem(sys.modules, "torch", fake_torch(None, False))
    fake_nvidia_smi(bare_machine, "NVIDIA GeForce RTX 3060, 551.23, 12288\n")

    env = device.collect_environment()

    assert env["torch"]["cuda_available"] is False
    assert any("CPU-only build" in w for w in env["warnings"])
    assert "none (CPU-only build)" in device.format_report(env)


def test_cuda_torch_without_usable_cuda_is_flagged(bare_machine: pytest.MonkeyPatch) -> None:
    bare_machine.setitem(sys.modules, "torch", fake_torch("12.1", False))
    env = device.collect_environment()
    assert any("built for CUDA 12.1 but cannot use CUDA" in w for w in env["warnings"])


def test_broken_torch_install_does_not_crash(bare_machine: pytest.MonkeyPatch) -> None:
    class BrokenTorchFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, name: str, path: Any, target: Any = None) -> None:
            if name == "torch":
                raise OSError("[WinError 126] fbgemm.dll could not be loaded")
            return None

    bare_machine.delitem(sys.modules, "torch")
    bare_machine.setattr(sys, "meta_path", [BrokenTorchFinder(), *sys.meta_path])

    env = device.collect_environment()

    assert env["torch"]["installed"] is True
    assert "fbgemm.dll" in env["torch"]["error"]
    assert any("did not load correctly" in w for w in env["warnings"])
    assert "failed to load" in device.format_report(env)


def test_failing_cuda_query_is_reported(bare_machine: pytest.MonkeyPatch) -> None:
    torch = fake_torch("12.1", True)

    def explode() -> bool:
        raise RuntimeError("CUDA driver initialization failed")

    torch.cuda.is_available = explode
    bare_machine.setitem(sys.modules, "torch", torch)

    info = device.probe_torch()
    assert "CUDA driver initialization failed" in info["error"]
    assert info["cuda_available"] is False


# --- ONNX Runtime ------------------------------------------------------------


def test_onnxruntime_cuda_provider_is_listed_but_not_claimed(bare_machine: pytest.MonkeyPatch) -> None:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    bare_machine.setitem(sys.modules, "onnxruntime", fake_onnxruntime(providers))

    env = device.collect_environment()
    text = device.format_report(env)

    assert env["onnxruntime"]["providers"] == providers
    assert "listed by this build; not verified" in text


def test_cpu_onnxruntime_on_a_gpu_machine_is_flagged(bare_machine: pytest.MonkeyPatch) -> None:
    bare_machine.setitem(sys.modules, "onnxruntime", fake_onnxruntime(["CPUExecutionProvider"]))
    fake_nvidia_smi(bare_machine, "NVIDIA T4, 535.104.05, 15360\n")

    env = device.collect_environment()

    assert any("does not offer CUDAExecutionProvider" in w for w in env["warnings"])
    assert "CUDA provider not available" in flat(device.format_report(env))


def test_conflicting_onnxruntime_packages_are_flagged(bare_machine: pytest.MonkeyPatch) -> None:
    versions = {"onnxruntime": "1.19.2", "onnxruntime-gpu": "1.19.2"}
    bare_machine.setattr(device, "_installed_version", versions.get)

    env = device.collect_environment()

    assert env["packages"]["ONNX Runtime"] == versions
    assert any("More than one ONNX Runtime package" in w for w in env["warnings"])


def test_conflicting_opencv_packages_are_flagged(bare_machine: pytest.MonkeyPatch) -> None:
    # e.g. Ultralytics pulls in opencv-python next to our opencv-python-headless
    versions = {"opencv-python": "4.14.0.94", "opencv-python-headless": "4.14.0.94"}
    bare_machine.setattr(device, "_installed_version", versions.get)

    env = device.collect_environment()

    assert env["packages"]["OpenCV"] == versions
    assert any("More than one OpenCV package" in w for w in env["warnings"])


# --- nvidia-smi --------------------------------------------------------------


def test_nvidia_smi_output_is_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_nvidia_smi(monkeypatch, "NVIDIA T4, 535.104.05, 15360\nNVIDIA A10G, 535.104.05, 23028\n")

    result = device.probe_nvidia_gpus()

    assert result["error"] is None
    assert result["gpus"] == [
        {"name": "NVIDIA T4", "driver_version": "535.104.05", "memory_total_mib": "15360"},
        {"name": "NVIDIA A10G", "driver_version": "535.104.05", "memory_total_mib": "23028"},
    ]


def test_failing_nvidia_smi_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_nvidia_smi(monkeypatch, "", returncode=9)
    result = device.probe_nvidia_gpus()
    assert result["gpus"] == []
    assert "exited with code 9" in result["error"]


def test_hanging_nvidia_smi_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device.shutil, "which", lambda name: "nvidia-smi")

    def run(command: list[str], **kwargs: Any) -> None:
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(device.subprocess, "run", run)
    assert "TimeoutExpired" in device.probe_nvidia_gpus()["error"]


# --- warnings logic on hand-made data ----------------------------------------


def _env(**overrides: Any) -> dict[str, Any]:
    env = {
        "python": {"version": "3.11.9"},
        "packages": {"ONNX Runtime": {}},
        "torch": {"installed": False},
        "onnxruntime": {"installed": False},
        "nvidia_gpus": {"nvidia_smi_found": False, "gpus": [], "error": None},
    }
    env.update(overrides)
    return env


def test_no_warnings_for_a_clean_cpu_setup() -> None:
    assert device.find_warnings(_env()) == []


def test_old_python_is_flagged() -> None:
    warnings = device.find_warnings(_env(python={"version": "3.10.12"}))
    assert warnings == ["Python 3.10.12 is older than the supported 3.11."]


# --- the real machine, nothing mocked ----------------------------------------


def test_report_runs_on_this_machine() -> None:
    """Whatever is (or is not) installed here, the report must not crash."""
    env = device.collect_environment()
    json.dumps(env)  # must be serialisable for saving alongside results
    assert "environment report" in device.format_report(env)
