"""Tests for edgevision.config.

Each test builds its config from the real configs/default.yaml and changes
one thing, so the tests stay in sync with the defaults instead of copying
them. Files are written to pytest's temporary directory (tmp_path).
"""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import Any

import pytest
import yaml

from edgevision.config import (
    SECTIONS,
    ConfigError,
    EdgeVisionConfig,
    config_to_dict,
    load_config,
    main,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture
def default_data() -> dict[str, Any]:
    """The default config as a plain dict that a test may modify."""
    return copy.deepcopy(yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8")))


def write_yaml(tmp_path: Path, data: Any, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def load_error(path: Path) -> str:
    with pytest.raises(ConfigError) as info:
        load_config(path)
    return str(info.value)


# --- valid configurations ----------------------------------------------------


def test_default_config_loads() -> None:
    config = load_config(DEFAULT_CONFIG)

    assert isinstance(config, EdgeVisionConfig)
    assert config.model.name == "yolov8n"
    assert config.model.input_size == 640
    assert isinstance(config.model.weights, Path)
    assert config.model.classes is None
    assert config.benchmark.scope == "detector"
    assert config.benchmark.warmup == 20 and config.benchmark.iterations == 300 and config.benchmark.runs == 3


def test_valid_config_from_temporary_file(tmp_path: Path, default_data: dict) -> None:
    default_data["model"]["classes"] = [0, 2]
    default_data["video"]["max_frames"] = None
    default_data["benchmark"]["scope"] = "pipeline"

    config = load_config(write_yaml(tmp_path, default_data))

    assert config.model.classes == (0, 2)
    assert config.video.max_frames is None
    assert config.benchmark.scope == "pipeline"


def test_load_config_accepts_string_path(default_data: dict, tmp_path: Path) -> None:
    assert load_config(str(write_yaml(tmp_path, default_data))).model.name == "yolov8n"


def test_integer_is_accepted_for_threshold_and_converted_to_float(
    tmp_path: Path, default_data: dict
) -> None:
    default_data["model"]["conf_threshold"] = 1
    config = load_config(write_yaml(tmp_path, default_data))
    assert config.model.conf_threshold == 1.0
    assert isinstance(config.model.conf_threshold, float)


def test_config_is_immutable() -> None:
    config = load_config(DEFAULT_CONFIG)
    with pytest.raises(FrozenInstanceError):
        config.model.input_size = 320  # type: ignore[misc]


def test_config_to_dict_round_trips(tmp_path: Path) -> None:
    original = load_config(DEFAULT_CONFIG)
    reloaded = load_config(write_yaml(tmp_path, config_to_dict(original)))
    assert reloaded == original


def test_schema_covers_every_top_level_field() -> None:
    assert list(SECTIONS) == [f.name for f in fields(EdgeVisionConfig)]


# --- unreadable or malformed files -------------------------------------------


def test_missing_file(tmp_path: Path) -> None:
    message = load_error(tmp_path / "does_not_exist.yaml")
    assert "not found" in message
    assert "does_not_exist.yaml" in message


def test_directory_instead_of_file(tmp_path: Path) -> None:
    assert "not a file" in load_error(tmp_path)


def test_malformed_yaml(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("model:\n  name: [unclosed\n", encoding="utf-8")
    message = load_error(path)
    assert "Malformed YAML" in message
    assert "line" in message  # PyYAML's message points at the problem position


def test_file_that_is_not_utf8(tmp_path: Path) -> None:
    path = tmp_path / "latin1.yaml"
    path.write_bytes("model:\n  name: caf\xe9\n".encode("latin-1"))
    assert "Could not read configuration file" in load_error(path)


def test_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert "is empty" in load_error(path)


def test_top_level_must_be_a_mapping(tmp_path: Path) -> None:
    assert "mapping of sections" in load_error(write_yaml(tmp_path, ["model", "video"]))


def test_duplicate_keys_are_rejected(tmp_path: Path) -> None:
    text = DEFAULT_CONFIG.read_text(encoding="utf-8").replace(
        "  input_size: 640", "  input_size: 640\n  input_size: 320"
    )
    path = tmp_path / "dupes.yaml"
    path.write_text(text, encoding="utf-8")
    assert "duplicate key 'input_size'" in load_error(path)


# --- missing and unknown fields ----------------------------------------------


def test_missing_section(tmp_path: Path, default_data: dict) -> None:
    del default_data["benchmark"]
    assert "missing required section 'benchmark'" in load_error(write_yaml(tmp_path, default_data))


def test_missing_field(tmp_path: Path, default_data: dict) -> None:
    del default_data["model"]["conf_threshold"]
    assert "missing required field 'model.conf_threshold'" in load_error(
        write_yaml(tmp_path, default_data)
    )


def test_section_must_be_a_mapping(tmp_path: Path, default_data: dict) -> None:
    default_data["video"] = "data/input/sample.mp4"
    assert "video must be a mapping of settings" in load_error(write_yaml(tmp_path, default_data))


def test_unknown_field_is_rejected(tmp_path: Path, default_data: dict) -> None:
    default_data["model"]["conf_treshold"] = 0.5  # typo of conf_threshold
    message = load_error(write_yaml(tmp_path, default_data))
    assert "unknown field 'model.conf_treshold'" in message
    assert "conf_threshold" in message  # the valid names are listed as a hint


def test_unknown_section_is_rejected(tmp_path: Path, default_data: dict) -> None:
    default_data["tensorrt"] = {"enabled": True}
    assert "unknown section 'tensorrt'" in load_error(write_yaml(tmp_path, default_data))


# --- wrong types and invalid values ------------------------------------------


@pytest.mark.parametrize(
    ("section", "key", "value", "expected_message"),
    [
        ("model", "name", "", "model.name must be a non-empty string"),
        ("model", "weights", 42, "model.weights must be a non-empty string"),
        ("model", "input_size", "640", "model.input_size must be an integer"),
        ("model", "input_size", 640.0, "model.input_size must be an integer"),
        ("model", "input_size", True, "model.input_size must be an integer"),
        ("model", "input_size", 641, "multiple of 32"),
        ("model", "input_size", 0, "model.input_size must be an integer >= 32"),
        ("model", "conf_threshold", 1.5, "model.conf_threshold must be a number between 0 and 1"),
        ("model", "iou_threshold", -0.1, "model.iou_threshold must be a number between 0 and 1"),
        ("model", "iou_threshold", "high", "model.iou_threshold must be a number between 0 and 1"),
        ("model", "classes", [], "model.classes must be null (all classes) or a non-empty list"),
        ("model", "classes", [0, -1], "model.classes must be null"),
        ("model", "classes", 0, "model.classes must be null"),
        ("export", "opset", 0, "export.opset must be an integer >= 1, or null"),
        ("export", "validate", "yes", "export.validate must be true or false"),
        ("video", "max_frames", 0, "video.max_frames must be an integer >= 1, or null"),
        ("tracking", "tracker", "sort", "tracking.tracker must be one of ['bytetrack']"),
        ("output", "overwrite", 1, "output.overwrite must be true or false"),
        ("benchmark", "scope", "model_only", "benchmark.scope must be one of ['detector', 'pipeline']"),
        ("benchmark", "device", "gpu", "benchmark.device must be one of ['auto', 'cpu', 'cuda']"),
        ("benchmark", "warmup", -1, "benchmark.warmup must be an integer >= 0"),
        ("benchmark", "iterations", 0, "benchmark.iterations must be an integer >= 1"),
        ("benchmark", "runs", 2.5, "benchmark.runs must be an integer >= 1"),
        ("benchmark", "preload_frames", 0, "benchmark.preload_frames must be an integer >= 1"),
    ],
)
def test_invalid_values_are_rejected(
    tmp_path: Path,
    default_data: dict,
    section: str,
    key: str,
    value: Any,
    expected_message: str,
) -> None:
    default_data[section][key] = value
    message = load_error(write_yaml(tmp_path, default_data))
    assert expected_message in message
    assert repr(value) in message  # the offending value is shown to the user


def test_all_problems_are_reported_together(tmp_path: Path, default_data: dict) -> None:
    default_data["model"]["conf_threshold"] = 2
    default_data["benchmark"]["runs"] = 0
    del default_data["video"]["input"]

    message = load_error(write_yaml(tmp_path, default_data))

    assert "3 problem(s)" in message
    assert "model.conf_threshold" in message
    assert "benchmark.runs" in message
    assert "video.input" in message


# --- command line ------------------------------------------------------------


def test_cli_accepts_valid_config(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(DEFAULT_CONFIG)]) == 0
    output = capsys.readouterr().out
    assert "Configuration OK" in output
    assert "input_size: 640" in output


def test_cli_reports_invalid_config(
    tmp_path: Path, default_data: dict, capsys: pytest.CaptureFixture[str]
) -> None:
    default_data["benchmark"]["device"] = "gpu"
    assert main([str(write_yaml(tmp_path, default_data))]) == 1
    error = capsys.readouterr().err
    assert error.startswith("ERROR: Invalid configuration")
    assert "benchmark.device" in error
