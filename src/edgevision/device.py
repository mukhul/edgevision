"""Report the hardware and software environment EdgeVision is running in.

Run it with::

    python -m edgevision.device          # human-readable report
    python -m edgevision.device --json   # machine-readable, for saving with results

Three different questions are deliberately kept apart:

1. **Hardware** - is an NVIDIA GPU present? Asked of the driver's own tool,
   ``nvidia-smi``, independently of any Python framework.
2. **Framework support** - can the installed PyTorch build use CUDA? Which
   execution providers does the installed ONNX Runtime build offer?
3. **Actual execution** - does a model really run on the GPU?

This module answers only 1 and 2. A GPU being present, or
``CUDAExecutionProvider`` being listed, does not prove that inference runs
on the GPU; that can only be confirmed by creating a session and running a
model, which later stages will do. Nothing is installed, downloaded, or
benchmarked here, and PyTorch / ONNX Runtime are optional.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from edgevision import __version__

logger = logging.getLogger(__name__)

MIN_PYTHON = (3, 11)
NVIDIA_SMI_TIMEOUT_S = 10.0

#: Optional packages: display name -> (possible pip distribution names, what it is used for).
#: Several distributions can provide the same module (e.g. onnxruntime / onnxruntime-gpu).
OPTIONAL_PACKAGES: dict[str, tuple[tuple[str, ...], str]] = {
    "PyTorch": (("torch",), "PyTorch inference backend"),
    "ONNX Runtime": (
        ("onnxruntime", "onnxruntime-gpu", "onnxruntime-directml", "onnxruntime-openvino"),
        "ONNX Runtime inference backend",
    ),
    "ONNX": (("onnx",), "ONNX export and validation"),
    "OpenCV": (
        (
            "opencv-python",
            "opencv-python-headless",
            "opencv-contrib-python",
            "opencv-contrib-python-headless",
        ),
        "video input and output",
    ),
    "Ultralytics": (("ultralytics",), "pretrained YOLO detector"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _import_optional(module: str) -> tuple[Any | None, str | None]:
    """Import an optional module without letting a failure crash the report.

    Returns ``(module, None)`` if it imported, ``(None, None)`` if it is not
    installed, and ``(None, error)`` if it is installed but broken.
    """
    try:
        return importlib.import_module(module), None
    except ImportError as exc:
        if getattr(exc, "name", None) == module:
            return None, None  # the package itself is missing
        return None, _error_text(exc)  # it exists but one of its own imports failed
    except Exception as exc:  # e.g. OSError when a CUDA DLL fails to load on Windows
        return None, _error_text(exc)


def _installed_version(distribution: str) -> str | None:
    """Version of an installed pip distribution, read from metadata without importing it."""
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Probes: each one answers a single question and never raises
# ---------------------------------------------------------------------------


def probe_python() -> dict[str, Any]:
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
    }


def probe_os() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "platform": platform.platform(),
    }


def _cpu_model_name() -> str | None:
    system = platform.system()
    try:
        if system == "Windows":
            import winreg

            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if system == "Linux":
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if system == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5, check=True,
            )
            return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError, IndexError) as exc:
        logger.debug("Could not read the CPU model name: %s", exc)
    return platform.processor() or None


def probe_cpu() -> dict[str, Any]:
    return {
        "model": _cpu_model_name(),
        "architecture": platform.machine(),
        "logical_cores": os.cpu_count(),
    }


def probe_nvidia_gpus() -> dict[str, Any]:
    """Detect NVIDIA GPUs through the driver tool, independently of PyTorch/ONNX Runtime."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"nvidia_smi_found": False, "gpus": [], "error": None}

    command = [
        executable,
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=NVIDIA_SMI_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"nvidia_smi_found": True, "gpus": [], "error": _error_text(exc)}
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        return {
            "nvidia_smi_found": True,
            "gpus": [],
            "error": f"nvidia-smi exited with code {result.returncode}: {detail}",
        }

    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.rsplit(",", 2)]
        if len(parts) == 3:
            name, driver, memory = parts
            gpus.append({"name": name, "driver_version": driver, "memory_total_mib": memory})
        elif line.strip():
            logger.debug("Ignoring unexpected nvidia-smi line: %r", line)
    return {"nvidia_smi_found": True, "gpus": gpus, "error": None}


def probe_packages() -> dict[str, dict[str, str]]:
    """Installed distributions (name -> version) for each optional package. Nothing is imported."""
    report: dict[str, dict[str, str]] = {}
    for display_name, (distributions, _purpose) in OPTIONAL_PACKAGES.items():
        found = {}
        for dist in distributions:
            version = _installed_version(dist)
            if version is not None:
                found[dist] = version
        report[display_name] = found
    return report


def probe_torch() -> dict[str, Any]:
    """What the installed PyTorch build can do. Imports torch only if it is installed."""
    torch, error = _import_optional("torch")
    if error is not None:
        return {"installed": True, "error": error}
    if torch is None:
        return {"installed": False}

    info: dict[str, Any] = {
        "installed": True,
        "version": str(torch.__version__),
        # None for CPU-only wheels; otherwise the CUDA version the wheel was built against.
        "cuda_build": torch.version.cuda,
        "cuda_available": False,
        "cuda_devices": [],
    }
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            count = torch.cuda.device_count()
            info["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(count)]
            info["cudnn_version"] = torch.backends.cudnn.version()
    except Exception as exc:  # report a broken CUDA setup instead of crashing
        info["error"] = f"CUDA query failed: {_error_text(exc)}"
    return info


def probe_onnxruntime() -> dict[str, Any]:
    """Execution providers offered by the installed ONNX Runtime build."""
    ort, error = _import_optional("onnxruntime")
    if error is not None:
        return {"installed": True, "error": error}
    if ort is None:
        return {"installed": False}
    try:
        providers = list(ort.get_available_providers())
    except Exception as exc:
        return {"installed": True, "version": str(ort.__version__), "error": _error_text(exc)}
    return {"installed": True, "version": str(ort.__version__), "providers": providers}


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------


def find_warnings(env: dict[str, Any]) -> list[str]:
    """Spot common setup problems in a collected environment."""
    warnings: list[str] = []

    version = env["python"]["version"]
    if tuple(int(p) for p in version.split(".")[:2]) < MIN_PYTHON:
        warnings.append(
            f"Python {version} is older than the supported {MIN_PYTHON[0]}.{MIN_PYTHON[1]}."
        )

    ort_distributions = env["packages"].get("ONNX Runtime", {})
    if len(ort_distributions) > 1:
        warnings.append(
            "More than one ONNX Runtime package is installed ("
            + ", ".join(ort_distributions)
            + "). They overwrite each other's files: uninstall all of them, then install exactly one."
        )

    opencv_distributions = env["packages"].get("OpenCV", {})
    if len(opencv_distributions) > 1:
        warnings.append(
            "More than one OpenCV package is installed ("
            + ", ".join(opencv_distributions)
            + "). They all provide the same cv2 module and overwrite each other: uninstall all "
            "of them, then install exactly one (EdgeVision uses opencv-python)."
        )

    torch, ort = env["torch"], env["onnxruntime"]
    for name, probe in (("PyTorch", torch), ("ONNX Runtime", ort)):
        if probe.get("error"):
            warnings.append(f"{name} is installed but did not load correctly: {probe['error']}")

    has_gpu = bool(env["nvidia_gpus"]["gpus"])
    torch_ok = torch.get("installed") and not torch.get("error")
    if torch_ok and torch.get("cuda_build") is None and has_gpu:
        warnings.append(
            "An NVIDIA GPU is present, but the installed PyTorch is a CPU-only build, "
            "so PyTorch cannot use the GPU."
        )
    if torch_ok and torch.get("cuda_build") and not torch.get("cuda_available"):
        warnings.append(
            f"PyTorch was built for CUDA {torch['cuda_build']} but cannot use CUDA here "
            "(no NVIDIA GPU, or the NVIDIA driver is missing or too old)."
        )

    ort_ok = ort.get("installed") and not ort.get("error")
    if ort_ok and has_gpu and "CUDAExecutionProvider" not in ort.get("providers", []):
        warnings.append(
            "An NVIDIA GPU is present, but the installed ONNX Runtime does not offer "
            "CUDAExecutionProvider (it is a CPU-only package)."
        )
    if env["nvidia_gpus"].get("error"):
        warnings.append(f"nvidia-smi was found but failed: {env['nvidia_gpus']['error']}")
    return warnings


def collect_environment() -> dict[str, Any]:
    """Gather the full environment report as JSON-serialisable data."""
    env: dict[str, Any] = {
        "edgevision_version": __version__,
        "python": probe_python(),
        "os": probe_os(),
        "cpu": probe_cpu(),
        "nvidia_gpus": probe_nvidia_gpus(),
        "packages": probe_packages(),
        "torch": probe_torch(),
        "onnxruntime": probe_onnxruntime(),
    }
    env["missing_optional_packages"] = [
        name for name, found in env["packages"].items() if not found
    ]
    env["warnings"] = find_warnings(env)
    return env


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------


def _row(label: str, value: Any) -> str:
    return f"  {label:<22}{value}"


def _torch_lines(torch: dict[str, Any]) -> list[str]:
    if not torch.get("installed"):
        return [_row("PyTorch", "not installed")]
    if "version" not in torch:
        return [_row("PyTorch", f"failed to load ({torch.get('error')})")]
    build = torch["cuda_build"] or "none (CPU-only build)"
    lines = [_row("PyTorch", torch["version"]), _row("  CUDA build", build)]
    if torch.get("cuda_available"):
        devices = ", ".join(torch["cuda_devices"]) or "no devices reported"
        lines.append(_row("  CUDA usable", f"yes - {devices}"))
        lines.append(_row("  cuDNN", torch.get("cudnn_version")))
    else:
        lines.append(_row("  CUDA usable", "no"))
    if torch.get("error"):
        lines.append(_row("  error", torch["error"]))
    return lines


def _onnxruntime_lines(ort: dict[str, Any]) -> list[str]:
    if not ort.get("installed"):
        return [_row("ONNX Runtime", "not installed")]
    if "providers" not in ort:
        return [_row("ONNX Runtime", f"failed to load ({ort.get('error')})")]
    providers = ort["providers"]
    lines = [
        _row("ONNX Runtime", ort["version"]),
        _row("  providers", ", ".join(providers) or "none"),
    ]
    if "CUDAExecutionProvider" in providers:
        lines.append(
            _row("  CUDA provider", "listed by this build; not verified (needs a real session)")
        )
    else:
        lines.append(_row("  CUDA provider", "not available"))
    return lines


def format_report(env: dict[str, Any]) -> str:
    """Render a collected environment as plain ASCII text (safe on any console)."""
    py, os_info, cpu, nvidia = env["python"], env["os"], env["cpu"], env["nvidia_gpus"]
    lines = [
        f"EdgeVision {env['edgevision_version']} - environment report",
        "",
        "System",
        _row("Python", f"{py['version']} ({py['implementation']}) {py['executable']}"),
        _row("OS", os_info["platform"]),
        _row(
            "CPU",
            f"{cpu['model'] or 'unknown model'} "
            f"({cpu['architecture'] or 'unknown arch'}, {cpu['logical_cores']} logical cores)",
        ),
        "",
        "GPU hardware (nvidia-smi, independent of Python frameworks)",
    ]
    if not nvidia["nvidia_smi_found"]:
        lines.append(_row("NVIDIA GPU", "none detected (nvidia-smi not found)"))
    elif nvidia["error"]:
        lines.append(_row("NVIDIA GPU", "unknown (nvidia-smi failed)"))
    elif not nvidia["gpus"]:
        lines.append(_row("NVIDIA GPU", "none reported by nvidia-smi"))
    for i, gpu in enumerate(nvidia["gpus"]):
        lines.append(
            _row(
                f"GPU {i}",
                f"{gpu['name']}, {gpu['memory_total_mib']} MiB, driver {gpu['driver_version']}",
            )
        )

    lines += ["", "Optional packages"]
    for name, found in env["packages"].items():
        purpose = OPTIONAL_PACKAGES[name][1]
        if found:
            installed = ", ".join(f"{dist} {ver}" for dist, ver in found.items())
            lines.append(_row(name, installed))
        else:
            lines.append(_row(name, f"not installed (needed for {purpose})"))

    lines += ["", "Framework acceleration support"]
    lines += _torch_lines(env["torch"])
    lines += _onnxruntime_lines(env["onnxruntime"])

    lines += ["", "Warnings"]
    lines += [f"  - {w}" for w in env["warnings"]] or ["  none"]

    lines += [
        "",
        "Note: this report checks hardware and installed frameworks only. It does not run",
        "a model, so it cannot show that GPU inference actually works.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m edgevision.device",
        description="Report the hardware and software environment (nothing is installed or run).",
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logging")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

    # Never crash on a console that cannot display a character (e.g. in a path).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    env = collect_environment()
    print(json.dumps(env, indent=2) if args.json else format_report(env))
    return 0


if __name__ == "__main__":
    sys.exit(main())
