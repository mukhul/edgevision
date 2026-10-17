"""Tests for scripts/detect_video.py.

Argument errors and --help run the script as a subprocess. Other tests call
main() in-process with the fake YOLO model from conftest.py and a small
synthetic video (needs OpenCV), so nothing is downloaded and no real model runs.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "detect_video.py"
CONFIG = REPO / "configs" / "default.yaml"
FRAMES_IN_VIDEO = 12


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("detect_video_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_subprocess(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False)


@pytest.fixture
def video(tmp_path: Path) -> Path:
    cv2 = pytest.importorskip("cv2", reason='OpenCV not installed (pip install -e ".[vision]")')
    np = pytest.importorskip("numpy")
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 120))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot encode MJPG video")
    try:
        for _ in range(FRAMES_IN_VIDEO):
            writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    finally:
        writer.release()
    return path


def custom_config(tmp_path: Path, **video_section: object) -> Path:
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data["video"].update(video_section)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# --- command-line interface ------------------------------------------------------------


def test_help_describes_the_options() -> None:
    result = run_subprocess("--help")
    assert result.returncode == 0
    for option in ("--max-frames", "--all-frames", "--model", "--device", "--conf", "--iou", "--json", "--overwrite"):
        assert option in result.stdout


@pytest.mark.parametrize(
    "args",
    [
        ["clip.mp4", "--max-frames", "0"],
        ["clip.mp4", "--max-frames", "5", "--all-frames"],  # mutually exclusive
        ["clip.mp4", "--conf", "2"],
        ["clip.mp4", "--device", "gpu"],
    ],
)
def test_invalid_arguments_exit_with_status_2(args: list[str]) -> None:
    result = run_subprocess(*args)
    assert result.returncode == 2
    assert "usage:" in result.stderr


def test_missing_video_fails_before_loading_the_model(
    fake_yolo, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = load_script().main([str(tmp_path / "missing.mp4"), "--config", str(CONFIG)])
    assert code == 1
    assert "Video not found" in capsys.readouterr().err
    assert fake_yolo.loaded_paths == []


# --- processing ------------------------------------------------------------------------


def test_processes_video_and_prints_summary(fake_yolo, video: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake_yolo.boxes = [(10, 10, 50, 60, 0.9, 15), (60, 10, 100, 80, 0.6, 0)]

    code = load_script().main([str(video), "--device", "cpu", "--config", str(CONFIG)])

    assert code == 0
    assert len(fake_yolo.loaded_paths) == 1  # the model is loaded once, not per frame
    assert len(fake_yolo.predict_calls) == FRAMES_IN_VIDEO
    out = " ".join(capsys.readouterr().out.split())  # ignore column alignment
    assert f"Frames processed {FRAMES_IN_VIDEO}" in out
    assert f"Total detections {2 * FRAMES_IN_VIDEO}" in out
    assert "not a benchmark" in out


def test_config_frame_limit_applies_by_default(fake_yolo, video: Path, tmp_path: Path) -> None:
    config = custom_config(tmp_path, max_frames=5)
    assert load_script().main([str(video), "--config", str(config)]) == 0
    assert len(fake_yolo.predict_calls) == 5


def test_max_frames_and_all_frames_override_the_config(fake_yolo, video: Path, tmp_path: Path) -> None:
    config = custom_config(tmp_path, max_frames=5)
    script = load_script()
    assert script.main([str(video), "--config", str(config), "--max-frames", "3"]) == 0
    assert len(fake_yolo.predict_calls) == 3
    assert script.main([str(video), "--config", str(config), "--all-frames"]) == 0
    assert len(fake_yolo.predict_calls) == 3 + FRAMES_IN_VIDEO


def test_video_path_defaults_to_config(fake_yolo, video: Path, tmp_path: Path) -> None:
    config = custom_config(tmp_path, input=str(video))
    assert load_script().main(["--config", str(config)]) == 0
    assert len(fake_yolo.predict_calls) == FRAMES_IN_VIDEO


# --- JSON output -------------------------------------------------------------------------


def test_writes_json_matching_the_summary(fake_yolo, video: Path, tmp_path: Path) -> None:
    fake_yolo.boxes = [(10, 10, 50, 60, 0.9, 15)]
    out = tmp_path / "results" / "clip.json"

    assert load_script().main([str(video), "--max-frames", "4", "--json", str(out), "--config", str(CONFIG)]) == 0

    doc = json.loads(out.read_text(encoding="utf-8"))  # Python's standard JSON parser
    assert doc["schema"] == "edgevision.video_detections"
    assert doc["video"]["width"] == 160 and doc["video"]["height"] == 120
    assert doc["detector"]["device"] == "cpu" and doc["detector"]["settings"]["conf_threshold"] == 0.25
    assert [f["frame_index"] for f in doc["frames"]] == [0, 1, 2, 3]
    assert doc["frames"][0]["detections"] == [
        {"bbox_xyxy": [10.0, 10.0, 50.0, 60.0], "confidence": 0.9, "class_id": 15, "class_name": "cat"}
    ]
    s = doc["summary"]
    assert s["frames_processed"] == 4 and s["total_detections"] == 4
    assert s["detections_by_class"] == {"cat": 4}
    assert s["stop_reason"] == "max_frames" and s["completed"] is True


def test_existing_json_is_not_overwritten_without_permission(
    fake_yolo, video: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "clip.json"
    out.write_text("previous results", encoding="utf-8")
    script = load_script()

    assert script.main([str(video), "--json", str(out), "--config", str(CONFIG)]) == 1
    assert "already exists" in capsys.readouterr().err
    assert out.read_text(encoding="utf-8") == "previous results"
    assert fake_yolo.loaded_paths == []  # refused before doing any work

    assert script.main([str(video), "--json", str(out), "--overwrite", "--config", str(CONFIG)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["summary"]["frames_processed"] == FRAMES_IN_VIDEO


def test_inference_failure_leaves_no_results_file(
    fake_yolo, video: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_yolo.predict_error = RuntimeError("kernel crashed")
    out = tmp_path / "clip.json"

    assert load_script().main([str(video), "--json", str(out), "--config", str(CONFIG)]) == 1

    err = capsys.readouterr().err
    assert "Detection failed on frame 0" in err and "kernel crashed" in err
    assert "No output files were written" in err
    assert not out.exists()
    assert not out.with_name(out.name + ".partial").exists()


def test_ctrl_c_leaves_no_results_file(
    fake_yolo, video: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_yolo.predict_error = KeyboardInterrupt()  # what Ctrl+C raises mid-inference
    out = tmp_path / "clip.json"

    assert load_script().main([str(video), "--json", str(out), "--config", str(CONFIG)]) == 130

    assert "Interrupted" in capsys.readouterr().err
    assert not out.exists()
    assert not out.with_name(out.name + ".partial").exists()


def test_cuda_request_without_cuda_fails(fake_yolo, video: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert load_script().main([str(video), "--device", "cuda", "--config", str(CONFIG)]) == 1
    assert "CUDA is not available" in capsys.readouterr().err


def test_undecodable_video_fails_clearly(fake_yolo, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("cv2", reason='OpenCV not installed (pip install -e ".[vision]")')
    broken = tmp_path / "broken.mp4"
    broken.write_text("not a video", encoding="utf-8")
    assert load_script().main([str(broken), "--config", str(CONFIG)]) == 1
    assert "ERROR:" in capsys.readouterr().err


# --- annotated output video (Prompt 5) ------------------------------------------------------


def reopen(path: Path) -> tuple[object, int]:
    from edgevision.video.reader import VideoReader

    with VideoReader(path) as reader:
        count = sum(1 for _ in reader)
        return reader.metadata, count


def test_writes_annotated_video_with_source_size_and_fps(
    fake_yolo, video: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_yolo.boxes = [(10, 10, 60, 70, 0.9, 15)]
    out = tmp_path / "annotated.mp4"

    assert load_script().main([str(video), "--output-video", str(out), "--config", str(CONFIG)]) == 0

    metadata, frames = reopen(out)
    assert frames == FRAMES_IN_VIDEO  # one annotated frame per processed frame
    assert (metadata.width, metadata.height) == (160, 120)  # same as the source
    assert metadata.fps == pytest.approx(10.0)  # the source's frame rate
    printed = " ".join(capsys.readouterr().out.split())
    assert f"Annotated video {out}" in printed and "(source)" in printed
    assert not (tmp_path / "annotated.partial.mp4").exists()


def test_annotated_video_respects_frame_limit_and_fps_override(fake_yolo, video: Path, tmp_path: Path) -> None:
    out = tmp_path / "annotated.mp4"
    args = [str(video), "--max-frames", "5", "--output-video", str(out), "--fps", "25", "--config", str(CONFIG)]
    assert load_script().main(args) == 0
    metadata, frames = reopen(out)
    assert frames == 5
    assert metadata.fps == pytest.approx(25.0)


def test_video_without_detections_is_still_written(fake_yolo, video: Path, tmp_path: Path) -> None:
    out = tmp_path / "annotated.mp4"  # fake_yolo returns no boxes by default
    assert load_script().main([str(video), "--output-video", str(out), "--config", str(CONFIG)]) == 0
    assert reopen(out)[1] == FRAMES_IN_VIDEO


def test_json_output_is_unchanged_when_also_writing_video(fake_yolo, video: Path, tmp_path: Path) -> None:
    fake_yolo.boxes = [(10, 10, 60, 70, 0.9, 15)]
    script = load_script()
    only_json, both = tmp_path / "only.json", tmp_path / "both.json"
    assert script.main([str(video), "--json", str(only_json), "--config", str(CONFIG)]) == 0
    assert script.main([str(video), "--json", str(both), "--output-video", str(tmp_path / "a.mp4"), "--config", str(CONFIG)]) == 0

    a, b = (json.loads(p.read_text(encoding="utf-8")) for p in (only_json, both))
    a.pop("created_utc"), b.pop("created_utc")
    assert a == b  # adding --output-video does not change the JSON


@pytest.mark.parametrize(
    "extra",
    [
        ["--fps", "25"],  # --fps without --output-video
        ["--output-video", "out.avi"],  # not .mp4
        ["--output-video", "x.mp4", "--fps", "0"],
        ["--output-video", "x.mp4", "--fps", "nan"],
    ],
)
def test_invalid_video_output_arguments(extra: list[str]) -> None:
    result = run_subprocess("clip.mp4", *extra)
    assert result.returncode == 2
    assert "usage:" in result.stderr


def test_output_video_cannot_be_the_input(fake_yolo, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    with pytest.raises(SystemExit) as info:
        load_script().main([str(source), "--output-video", str(tmp_path / "." / "clip.mp4"), "--config", str(CONFIG)])
    assert info.value.code == 2
    assert "must not be the input video" in capsys.readouterr().err


def test_output_video_and_json_must_differ(fake_yolo, video: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    same = tmp_path / "out.mp4"
    with pytest.raises(SystemExit):
        load_script().main([str(video), "--output-video", str(same), "--json", str(same), "--config", str(CONFIG)])
    assert "must be different files" in capsys.readouterr().err


def test_existing_output_video_is_protected(fake_yolo, video: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "annotated.mp4"
    out.write_bytes(b"previous video")
    assert load_script().main([str(video), "--output-video", str(out), "--config", str(CONFIG)]) == 1
    assert "already exists" in capsys.readouterr().err
    assert out.read_bytes() == b"previous video"
    assert fake_yolo.loaded_paths == []


def test_failure_mid_video_leaves_no_output_video(
    fake_yolo, video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = {"n": 0}
    script = load_script()
    real_annotate = script.annotate_frame

    def failing_annotate(image, detections):  # fail on the 4th frame, after 3 were written
        calls["n"] += 1
        if calls["n"] == 4:
            raise OSError("disk full")
        return real_annotate(image, detections)

    monkeypatch.setattr(script, "annotate_frame", failing_annotate)
    out, js = tmp_path / "annotated.mp4", tmp_path / "out.json"

    assert script.main([str(video), "--output-video", str(out), "--json", str(js), "--config", str(CONFIG)]) == 1

    assert "disk full" in capsys.readouterr().err
    for path in (out, tmp_path / "annotated.partial.mp4", js, tmp_path / "out.json.partial"):
        assert not path.exists(), path


def test_inference_failure_leaves_no_output_video(fake_yolo, video: Path, tmp_path: Path) -> None:
    fake_yolo.predict_error = RuntimeError("kernel crashed")
    out = tmp_path / "annotated.mp4"
    assert load_script().main([str(video), "--output-video", str(out), "--config", str(CONFIG)]) == 1
    assert not out.exists() and not (tmp_path / "annotated.partial.mp4").exists()


def test_ctrl_c_leaves_no_output_video(fake_yolo, video: Path, tmp_path: Path) -> None:
    fake_yolo.predict_error = KeyboardInterrupt()
    out = tmp_path / "annotated.mp4"
    assert load_script().main([str(video), "--output-video", str(out), "--config", str(CONFIG)]) == 130
    assert not out.exists() and not (tmp_path / "annotated.partial.mp4").exists()


def test_odd_sized_source_fails_clearly(fake_yolo, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cv2 = pytest.importorskip("cv2", reason='OpenCV not installed (pip install -e ".[vision]")')
    np = pytest.importorskip("numpy")
    odd = tmp_path / "odd.avi"
    writer = cv2.VideoWriter(str(odd), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    for _ in range(3):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    script = load_script()

    # OpenCV cannot write an odd-sized test video (it silently crops it, the very
    # problem being tested), so wrap the pipeline to hand out 63-px-wide frames.
    class OddFrames:
        def __init__(self, pipeline):
            self.pipeline = pipeline

        def __getattr__(self, name):
            return getattr(self.pipeline, name)

        def __enter__(self):
            self.pipeline.__enter__()
            return self

        def __exit__(self, *exc):
            return self.pipeline.__exit__(*exc)

        def __iter__(self):
            for result in self.pipeline:
                yield type(result)(result.frame_index, result.detections, image=result.image[:, :63].copy())

    real_pipeline = script.VideoDetectionPipeline
    script.VideoDetectionPipeline = lambda *a, **k: OddFrames(real_pipeline(*a, **k))
    out = tmp_path / "odd.mp4"

    assert script.main([str(odd), "--output-video", str(out), "--config", str(CONFIG)]) == 1
    assert "even width and height" in capsys.readouterr().err
    assert not out.exists()


# --- tracking (Prompt 6) ----------------------------------------------------------------


@pytest.fixture
def script_with_fake_tracker(fake_tracker_class):
    """The script module with ByteTrackTracker replaced by the deterministic fake."""
    script = load_script()
    created = []

    def make(config=None):
        tracker = fake_tracker_class(config)
        created.append(tracker)
        return tracker

    script.ByteTrackTracker = make
    script.created_trackers = created
    return script


def test_tracking_adds_track_ids_to_json_and_summary(
    fake_yolo, video: Path, tmp_path: Path, script_with_fake_tracker, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_yolo.boxes = [(10, 10, 50, 60, 0.9, 15), (60, 10, 100, 80, 0.6, 0)]
    out = tmp_path / "tracks.json"
    args = [str(video), "--tracking", "--max-frames", "4", "--json", str(out), "--config", str(CONFIG)]

    assert script_with_fake_tracker.main(args) == 0

    tracker = script_with_fake_tracker.created_trackers[0]
    assert tracker.reset_calls == 1 and tracker.update_calls == [0, 1, 2, 3]
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["tracker"]["name"] == "fake"
    assert [[d["track_id"] for d in f["detections"]] for f in doc["frames"]] == [[1, 2]] * 4
    assert doc["summary"]["tracking"]["distinct_track_ids_observed"] == 2
    printed = " ".join(capsys.readouterr().out.split())
    assert "Distinct track IDs 2 observed in this video" in printed
    assert "not a count of real objects" in printed
    assert fake_yolo.predict_calls and len(fake_yolo.predict_calls) == 4  # inference once per frame


def test_tracking_is_off_by_default(fake_yolo, video: Path, tmp_path: Path, script_with_fake_tracker) -> None:
    out = tmp_path / "plain.json"
    assert script_with_fake_tracker.main([str(video), "--json", str(out), "--config", str(CONFIG)]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert script_with_fake_tracker.created_trackers == []
    assert "tracker" not in doc and "tracking" not in doc["summary"]


def test_config_can_enable_tracking_and_cli_can_disable_it(
    fake_yolo, video: Path, tmp_path: Path, script_with_fake_tracker
) -> None:
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data["tracking"]["enabled"] = True
    config = tmp_path / "tracking_on.yaml"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")

    assert script_with_fake_tracker.main([str(video), "--config", str(config)]) == 0
    assert len(script_with_fake_tracker.created_trackers) == 1
    assert script_with_fake_tracker.main([str(video), "--no-tracking", "--config", str(config)]) == 0
    assert len(script_with_fake_tracker.created_trackers) == 1  # no new tracker created


def test_tracker_config_is_passed_through(fake_yolo, video: Path, tmp_path: Path, script_with_fake_tracker) -> None:
    custom = tmp_path / "my_bytetrack.yaml"
    args = [str(video), "--tracking", "--tracker-config", str(custom), "--config", str(CONFIG)]
    assert script_with_fake_tracker.main(args) == 0
    assert script_with_fake_tracker.created_trackers[0].config_source == str(custom)


def test_tracker_config_without_tracking_is_an_argument_error() -> None:
    result = run_subprocess("clip.mp4", "--tracker-config", "x.yaml")
    assert result.returncode == 2
    assert "--tracker-config only applies" in result.stderr


def test_tracker_setup_failure_fails_before_loading_the_model(
    fake_yolo, video: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from edgevision.optional import MissingDependencyError

    script = load_script()

    def missing(config=None):
        raise MissingDependencyError('lap is required to run ByteTrack tracking but is not installed. Install it with: python -m pip install -e ".[tracking]"')

    script.ByteTrackTracker = missing
    assert script.main([str(video), "--tracking", "--config", str(CONFIG)]) == 1
    assert '.[tracking]' in capsys.readouterr().err
    assert fake_yolo.loaded_paths == []


def test_tracking_with_annotated_video(fake_yolo, video: Path, tmp_path: Path, script_with_fake_tracker) -> None:
    fake_yolo.boxes = [(10, 10, 50, 60, 0.9, 15)]
    out = tmp_path / "tracked.mp4"
    args = [str(video), "--tracking", "--output-video", str(out), "--config", str(CONFIG)]
    assert script_with_fake_tracker.main(args) == 0
    assert reopen(out)[1] == FRAMES_IN_VIDEO


def test_tracking_failure_mid_video_leaves_no_outputs(
    fake_yolo, video: Path, tmp_path: Path, script_with_fake_tracker, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_yolo.boxes = [(10, 10, 50, 60, 0.9, 15)]
    original = script_with_fake_tracker.ByteTrackTracker

    def failing(config=None):
        tracker = original(config)
        tracker.fail_at_frame = 3
        return tracker

    script_with_fake_tracker.ByteTrackTracker = failing
    js, mp4 = tmp_path / "t.json", tmp_path / "t.mp4"
    args = [str(video), "--tracking", "--json", str(js), "--output-video", str(mp4), "--config", str(CONFIG)]

    assert script_with_fake_tracker.main(args) == 1
    assert "Tracking failed on frame 3" in capsys.readouterr().err
    assert not js.exists() and not mp4.exists()
