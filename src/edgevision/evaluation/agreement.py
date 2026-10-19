"""Compare the detections of two runs on the same video (no ground truth needed).

Inputs are ``detect_video.py --json`` files (schema ``edgevision.video_detections``
v1). The comparison says how similar two outputs are, e.g. PyTorch vs ONNX
Runtime on the same clip. It does **not** say which is correct: without
labelled ground truth, a detection present in only one output may be a
true object the other missed or a false alarm the other avoided.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from edgevision.evaluation.detection_metrics import agreement
from edgevision.models.base import Detection

VIDEO_SCHEMA = "edgevision.video_detections"


class DetectionsFileError(ValueError):
    """A detections file is missing, malformed or not comparable."""


def load_video_detections(path: str | os.PathLike[str]) -> tuple[dict[str, Any], list[list[Detection]]]:
    """The file's header and its detections as ``Detection`` records, one list per frame in order."""
    path = Path(path)
    if not path.is_file():
        raise DetectionsFileError(f"Detections file not found: {path}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DetectionsFileError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("schema") != VIDEO_SCHEMA or doc.get("schema_version") != 1:
        raise DetectionsFileError(f"{path} is not a detect_video.py --json result ({VIDEO_SCHEMA} v1)")
    frames = doc.get("frames")
    if not isinstance(frames, list) or not frames:
        raise DetectionsFileError(f"{path} contains no frames")
    out: list[list[Detection]] = []
    for i, frame in enumerate(frames):
        if not isinstance(frame, dict) or frame.get("frame_index") != i or not isinstance(frame.get("detections"), list):
            raise DetectionsFileError(f"{path}: frames[{i}] must have frame_index {i} and a detections list")
        try:
            out.append([Detection(i, tuple(d["bbox_xyxy"]), d["confidence"], d["class_id"], d["class_name"])
                        for d in frame["detections"]])
        except (KeyError, TypeError, ValueError) as exc:
            raise DetectionsFileError(f"{path}: invalid detection in frame {i}: {exc}") from exc
    return doc, out


def comparison_problems(doc_a: dict[str, Any], doc_b: dict[str, Any], frames_a: int, frames_b: int) -> list[str]:
    """Differences that make the two files describe different work (each weakens the comparison)."""
    problems = []
    if frames_a != frames_b:
        problems.append(f"different number of frames: {frames_a} vs {frames_b}")
    video_a, video_b = doc_a.get("video", {}), doc_b.get("video", {})
    for key in ("path", "width", "height"):
        if video_a.get(key) != video_b.get(key):
            problems.append(f"video {key} differs: {video_a.get(key)!r} vs {video_b.get(key)!r}")
    settings_a = doc_a.get("detector", {}).get("settings", {})
    settings_b = doc_b.get("detector", {}).get("settings", {})
    for key in sorted(set(settings_a) | set(settings_b)):
        if settings_a.get(key) != settings_b.get(key):
            problems.append(f"setting {key} differs: {settings_a.get(key)!r} vs {settings_b.get(key)!r}")
    return problems


def letterbox_of(doc: dict[str, Any]) -> str | None:
    """How a run padded its frames: as recorded, else the backend's fixed behaviour in files written
    before the padding was recorded (PyTorch 'minimal', ONNX 'square'); None if unknown."""
    detector = doc.get("detector", {})
    if isinstance(detector.get("letterbox"), str):
        return detector["letterbox"]
    return {"pytorch": "minimal", "onnx": "square"}.get(detector.get("backend"))


def compare_files(path_a: str | os.PathLike[str], path_b: str | os.PathLike[str], iou_threshold: float = 0.5) -> dict:
    doc_a, frames_a = load_video_detections(path_a)
    doc_b, frames_b = load_video_detections(path_b)
    problems = comparison_problems(doc_a, doc_b, len(frames_a), len(frames_b))
    if len(frames_a) != len(frames_b):
        raise DetectionsFileError("; ".join(problems) + ": the files do not cover the same frames")
    result = agreement(frames_a, frames_b, iou_threshold=iou_threshold)
    notes = []
    padding = {letterbox_of(doc_a), letterbox_of(doc_b)}
    if padding == {"minimal", "square"}:
        notes.append("One run padded frames only to a multiple of 32 (PyTorch's default, e.g. 640x384 for 1024x576), "
                     "the other to the full square input (e.g. 640x640). The network saw slightly different images, "
                     "which explains most differences between these two outputs. For a like-for-like comparison, "
                     "run PyTorch with --pytorch-letterbox square.")
    return {
        "a": {"file": Path(path_a).name, **{k: doc_a.get("detector", {}).get(k) for k in ("backend", "model", "device")},
              "letterbox": letterbox_of(doc_a)},
        "b": {"file": Path(path_b).name, **{k: doc_b.get("detector", {}).get(k) for k in ("backend", "model", "device")},
              "letterbox": letterbox_of(doc_b)},
        "comparability_problems": problems,
        "agreement": result,
        "notes": notes,
    }


def format_comparison(report: dict[str, Any]) -> str:
    a, b, r = report["a"], report["b"], report["agreement"]

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{100 * value:.1f}%"

    iou, conf = r["matched_iou"], r["matched_confidence_abs_diff"]
    lines = [
        "Detection agreement between two runs (NOT an accuracy measure: there is no ground truth)",
        f"  A  {a['file']}: {a['backend']} ({a['model']}, {a['device']}, padding {a.get('letterbox') or 'unknown'})",
        f"  B  {b['file']}: {b['backend']} ({b['model']}, {b['device']}, padding {b.get('letterbox') or 'unknown'})",
        f"  Frames compared        {r['frames']} (match = same class and IoU >= {r['iou_threshold']})",
        f"  Detections             A {r['detections_a']}, B {r['detections_b']}",
        f"  Matched pairs          {r['matched']}  ({pct(r['matched_share'])} of all distinct detections)",
        f"  Only in A / only in B  {r['only_a']} / {r['only_b']}",
        f"  Frames fully matched   {r['frames_with_all_detections_matched']} of {r['frames']}",
    ]
    if iou:
        lines.append(f"  IoU of matched boxes   median {iou['median']:.3f}, mean {iou['mean']:.3f}, min {iou['min']:.3f}")
    if conf:
        lines.append(f"  |confidence A - B|     median {conf['median']:.3f}, mean {conf['mean']:.3f}, max {conf['max']:.3f}")
    lines += ["", f"  {'class':<16}{'matched':>8}{'only A':>8}{'only B':>8}"]
    for name, c in r["per_class"].items():
        lines.append(f"  {name:<16}{c['matched']:>8}{c['only_a']:>8}{c['only_b']:>8}")
    if report["comparability_problems"]:
        lines += ["", "Not directly comparable - these differ:"] + [f"  - {p}" for p in report["comparability_problems"]]
    lines += [f"\nNote: {n}" for n in report["notes"]]
    lines.append("\nA detection found by only one run may be a real object the other missed, or a false alarm the "
                 "other avoided; only labelled ground truth can tell which.")
    return "\n".join(lines)
