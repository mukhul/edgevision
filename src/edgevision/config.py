"""Load and validate EdgeVision experiment configuration files.

Default values live in exactly one place: ``configs/default.yaml``. This
module defines the *shape* a configuration file must have and checks every
field, so later stages can trust the settings without re-checking them.

Validation is strict:

* every section and field must be present (use ``null`` for "not set"),
* unknown sections and fields are rejected, which catches typos such as
  ``conf_treshold`` that would otherwise be silently ignored,
* duplicate keys in the YAML file are rejected,
* all problems are reported together, not one at a time.

Check a file from the command line::

    python -m edgevision.config                      # configs/default.yaml
    python -m edgevision.config path/to/other.yaml
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

#: Default config location, relative to the repository root. Commands are
#: expected to be run from the repository root.
DEFAULT_CONFIG_PATH = Path("configs") / "default.yaml"

BACKENDS = ("pytorch", "onnx_cpu", "onnx_cuda")
DEVICES = ("auto", "cpu", "cuda")
SCOPES = ("model_only", "end_to_end")
TRACKERS = ("bytetrack",)
PRECISIONS = ("fp32", "fp16")
#: YOLO models downsample by 32, so the square input size must be a multiple of it.
MODEL_STRIDE = 32


class ConfigError(Exception):
    """Raised when a configuration file cannot be read or is invalid."""


# ---------------------------------------------------------------------------
# Typed configuration objects
# ---------------------------------------------------------------------------
# These dataclasses have no default values on purpose: defaults belong in
# configs/default.yaml. ``frozen=True`` stops code from changing settings
# half-way through an experiment.


@dataclass(frozen=True)
class ModelConfig:
    name: str
    weights: Path
    onnx_path: Path
    input_size: int
    conf_threshold: float
    iou_threshold: float
    classes: tuple[int, ...] | None  # None means "all classes"


@dataclass(frozen=True)
class ExportConfig:
    opset: int | None  # None means "not chosen yet"
    dynamic_shapes: bool
    validate: bool


@dataclass(frozen=True)
class VideoConfig:
    input: Path
    max_frames: int | None  # None means "the whole video"


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool
    tracker: str


@dataclass(frozen=True)
class OutputConfig:
    dir: Path
    overwrite: bool


@dataclass(frozen=True)
class BenchmarkConfig:
    backends: tuple[str, ...]
    device: str
    scopes: tuple[str, ...]
    warmup: int
    frames: int
    runs: int
    batch_size: int
    precision: str


@dataclass(frozen=True)
class EdgeVisionConfig:
    model: ModelConfig
    export: ExportConfig
    video: VideoConfig
    tracking: TrackingConfig
    output: OutputConfig
    benchmark: BenchmarkConfig


#: Section name -> dataclass. The expected field names come from the
#: dataclasses themselves, so the schema is never written down twice.
SECTIONS: dict[str, type] = {
    "model": ModelConfig,
    "export": ExportConfig,
    "video": VideoConfig,
    "tracking": TrackingConfig,
    "output": OutputConfig,
    "benchmark": BenchmarkConfig,
}


# ---------------------------------------------------------------------------
# YAML reading
# ---------------------------------------------------------------------------


class _StrictSafeLoader(yaml.SafeLoader):
    """A SafeLoader that refuses duplicate keys instead of keeping the last one."""


def _construct_mapping_without_duplicates(
    loader: _StrictSafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark
            )
        seen.add(key)
    return loader.construct_mapping(node, deep=deep)


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping_without_duplicates
)


def _read_yaml(path: Path) -> Any:
    if not path.exists():
        raise ConfigError(
            f"Configuration file not found: {path} (looked relative to {Path.cwd()}). "
            "Run commands from the repository root or pass an explicit path."
        )
    if not path.is_file():
        raise ConfigError(f"Configuration path is not a file: {path}")
    try:
        # Passing the open file (not a string) lets PyYAML name the file in errors.
        with path.open(encoding="utf-8") as stream:
            return yaml.load(stream, Loader=_StrictSafeLoader)  # SafeLoader: never runs code
    except yaml.YAMLError as exc:
        raise ConfigError(f"Malformed YAML in {path}:\n{exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Could not read configuration file {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _describe(value: Any) -> str:
    return f"{value!r} ({type(value).__name__})"


def _is_int(value: Any) -> bool:
    # bool is a subclass of int in Python, so `True` would otherwise pass as 1.
    return isinstance(value, int) and not isinstance(value, bool)


class _Validator:
    """Checks fields one at a time and collects every problem it finds."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def fail(self, key: str, expected: str, value: Any) -> None:
        self.errors.append(f"{key} must be {expected}, got {_describe(value)}")

    def section(self, root: dict[str, Any], name: str) -> dict[str, Any] | None:
        if name not in root:
            self.errors.append(f"missing required section '{name}'")
            return None
        data = root[name]
        if not isinstance(data, dict):
            self.fail(name, "a mapping of settings", data)
            return None
        expected = [f.name for f in fields(SECTIONS[name])]
        for key in expected:
            if key not in data:
                self.errors.append(f"missing required field '{name}.{key}'")
        for key in data:
            if key not in expected:
                self.errors.append(
                    f"unknown field '{name}.{key}' (expected one of: {', '.join(expected)})"
                )
        return data

    # Each check below receives the section dict and the field name. A field
    # that is absent has already been reported by section(), so it is skipped.

    def text(self, data: dict[str, Any], section: str, key: str) -> Any:
        value = data.get(key)
        if key in data and not (isinstance(value, str) and value.strip()):
            self.fail(f"{section}.{key}", "a non-empty string", value)
        return value

    def path(self, data: dict[str, Any], section: str, key: str) -> Any:
        value = self.text(data, section, key)
        return Path(value) if isinstance(value, str) else value

    def flag(self, data: dict[str, Any], section: str, key: str) -> Any:
        value = data.get(key)
        if key in data and not isinstance(value, bool):
            self.fail(f"{section}.{key}", "true or false", value)
        return value

    def integer(
        self,
        data: dict[str, Any],
        section: str,
        key: str,
        *,
        minimum: int,
        nullable: bool = False,
        multiple_of: int | None = None,
    ) -> Any:
        if key not in data:
            return None
        value = data[key]
        if value is None and nullable:
            return None
        ok = _is_int(value) and value >= minimum
        if ok and multiple_of is not None:
            ok = value % multiple_of == 0
        if not ok:
            expected = f"an integer >= {minimum}"
            if multiple_of is not None:
                expected += f" and a multiple of {multiple_of}"
            if nullable:
                expected += ", or null"
            self.fail(f"{section}.{key}", expected, value)
        return value

    def fraction(self, data: dict[str, Any], section: str, key: str) -> Any:
        if key not in data:
            return None
        value = data[key]
        if (_is_int(value) or isinstance(value, float)) and 0.0 <= value <= 1.0:
            return float(value)
        self.fail(f"{section}.{key}", "a number between 0 and 1", value)
        return value

    def choice(self, data: dict[str, Any], section: str, key: str, choices: tuple[str, ...]) -> Any:
        value = data.get(key)
        if key in data and value not in choices:
            self.fail(f"{section}.{key}", f"one of {list(choices)}", value)
        return value

    def choice_list(
        self, data: dict[str, Any], section: str, key: str, choices: tuple[str, ...]
    ) -> Any:
        if key not in data:
            return None
        value = data[key]
        if (
            isinstance(value, list)
            and value
            and all(isinstance(item, str) and item in choices for item in value)
            and len(set(value)) == len(value)
        ):
            return tuple(value)
        self.fail(f"{section}.{key}", f"a non-empty list without repeats, taken from {list(choices)}", value)
        return value

    def class_ids(self, data: dict[str, Any], section: str, key: str) -> Any:
        if key not in data:
            return None
        value = data[key]
        if value is None:
            return None
        if isinstance(value, list) and value and all(_is_int(i) and i >= 0 for i in value):
            return tuple(value)
        self.fail(
            f"{section}.{key}",
            "null (all classes) or a non-empty list of non-negative integer class IDs",
            value,
        )
        return value


def _validate(root: Any, source: str) -> EdgeVisionConfig:
    if root is None:
        raise ConfigError(f"Configuration file {source} is empty")
    if not isinstance(root, dict):
        raise ConfigError(
            f"Configuration file {source} must contain a mapping of sections at the top level, "
            f"got {type(root).__name__}"
        )

    v = _Validator()
    for name in root:
        if name not in SECTIONS:
            v.errors.append(f"unknown section '{name}' (expected one of: {', '.join(SECTIONS)})")

    values: dict[str, dict[str, Any]] = {}

    if (d := v.section(root, "model")) is not None:
        values["model"] = {
            "name": v.text(d, "model", "name"),
            "weights": v.path(d, "model", "weights"),
            "onnx_path": v.path(d, "model", "onnx_path"),
            "input_size": v.integer(d, "model", "input_size", minimum=MODEL_STRIDE, multiple_of=MODEL_STRIDE),
            "conf_threshold": v.fraction(d, "model", "conf_threshold"),
            "iou_threshold": v.fraction(d, "model", "iou_threshold"),
            "classes": v.class_ids(d, "model", "classes"),
        }
    if (d := v.section(root, "export")) is not None:
        values["export"] = {
            "opset": v.integer(d, "export", "opset", minimum=1, nullable=True),
            "dynamic_shapes": v.flag(d, "export", "dynamic_shapes"),
            "validate": v.flag(d, "export", "validate"),
        }
    if (d := v.section(root, "video")) is not None:
        values["video"] = {
            "input": v.path(d, "video", "input"),
            "max_frames": v.integer(d, "video", "max_frames", minimum=1, nullable=True),
        }
    if (d := v.section(root, "tracking")) is not None:
        values["tracking"] = {
            "enabled": v.flag(d, "tracking", "enabled"),
            "tracker": v.choice(d, "tracking", "tracker", TRACKERS),
        }
    if (d := v.section(root, "output")) is not None:
        values["output"] = {
            "dir": v.path(d, "output", "dir"),
            "overwrite": v.flag(d, "output", "overwrite"),
        }
    if (d := v.section(root, "benchmark")) is not None:
        values["benchmark"] = {
            "backends": v.choice_list(d, "benchmark", "backends", BACKENDS),
            "device": v.choice(d, "benchmark", "device", DEVICES),
            "scopes": v.choice_list(d, "benchmark", "scopes", SCOPES),
            "warmup": v.integer(d, "benchmark", "warmup", minimum=0),
            "frames": v.integer(d, "benchmark", "frames", minimum=1),
            "runs": v.integer(d, "benchmark", "runs", minimum=1),
            "batch_size": v.integer(d, "benchmark", "batch_size", minimum=1),
            "precision": v.choice(d, "benchmark", "precision", PRECISIONS),
        }

    if v.errors:
        problems = "\n".join(f"  - {e}" for e in v.errors)
        raise ConfigError(f"Invalid configuration {source} ({len(v.errors)} problem(s)):\n{problems}")

    return EdgeVisionConfig(**{name: SECTIONS[name](**values[name]) for name in SECTIONS})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> EdgeVisionConfig:
    """Read, validate and return the configuration stored at ``path``.

    Raises:
        ConfigError: if the file is missing, unreadable, not valid YAML, or
            does not match the expected structure. The message lists every
            problem found.
    """
    path = Path(path)
    return _validate(_read_yaml(path), source=str(path))


def config_to_dict(config: EdgeVisionConfig) -> dict[str, dict[str, Any]]:
    """Convert a config into plain YAML/JSON-friendly types (for saving with results)."""

    def plain(value: Any) -> Any:
        if isinstance(value, Path):
            return value.as_posix()
        if isinstance(value, tuple):
            return list(value)
        return value

    return {
        name: {f.name: plain(getattr(section, f.name)) for f in fields(section)}
        for name, section in ((n, getattr(config, n)) for n in SECTIONS)
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m edgevision.config",
        description="Validate an EdgeVision configuration file and print the resolved settings.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"configuration file to check (default: {DEFAULT_CONFIG_PATH.as_posix()})",
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.path)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Configuration OK: {args.path}\n")
    print(yaml.safe_dump(config_to_dict(config), sort_keys=False), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
