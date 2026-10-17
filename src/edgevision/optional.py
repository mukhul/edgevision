"""Import optional dependencies with actionable error messages.

The base package needs only PyYAML. Heavier packages (OpenCV, PyTorch,
Ultralytics) are optional "extras" and are imported only when a feature that
needs them is actually used.
"""

from __future__ import annotations

import importlib
from types import ModuleType


class MissingDependencyError(ImportError):
    """An optional dependency needed for this feature is missing or broken."""


def install_hint(extra: str) -> str:
    return f'python -m pip install -e ".[{extra}]"'


def import_optional(module: str, *, display_name: str, extra: str, purpose: str) -> ModuleType:
    """Import ``module`` or raise :class:`MissingDependencyError` explaining how to install it.

    Args:
        module: Import name, e.g. ``"cv2"``.
        display_name: Human-readable name, e.g. ``"OpenCV"``.
        extra: The pyproject extra that installs it, e.g. ``"vision"``.
        purpose: What it is needed for, completing "required to ...".
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        if getattr(exc, "name", None) == module:
            raise MissingDependencyError(
                f"{display_name} is required to {purpose} but is not installed. "
                f"Install it with: {install_hint(extra)}"
            ) from exc
        raise MissingDependencyError(
            f"{display_name} is installed but could not be imported ({exc}). "
            f"Try reinstalling it with: {install_hint(extra)}"
        ) from exc
