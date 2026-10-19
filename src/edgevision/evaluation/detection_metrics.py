"""Detection metrics: quality against ground truth, and agreement between two backends.

**Against ground truth** (needs labelled images, see ``annotations.py``):

* A prediction is a *true positive* (TP) at IoU threshold ``t`` when it is
  matched to a ground-truth box of the **same class** with IoU >= ``t``.
  Within each image, predictions are matched from the highest confidence
  down; each ground-truth box can be matched once, to the unmatched box with
  the highest IoU. Predictions that match nothing are *false positives* (FP),
  except those lying inside a "crowd" region (COCO ``iscrowd``; at least
  ``t`` of the prediction's area inside the region, as in COCO), which are
  ignored. Ground-truth boxes left unmatched are *false negatives* (FN).
* Precision = TP / (TP + FP), recall = TP / (TP + FN), F1 = their harmonic
  mean, computed on all predictions given (i.e. at the detector's confidence
  threshold), at IoU 0.50.
* Average precision (AP) per class follows the COCO definition: predictions
  of the class from all images sorted by confidence, precision made
  monotonically decreasing, then sampled at 101 recall points (0, 0.01 ...
  1) and averaged. mAP@0.5 is the mean over classes that have ground truth;
  mAP@0.5:0.95 additionally averages over IoU 0.50, 0.55 ... 0.95.

**Between two backends** (no ground truth): :func:`agreement` matches the
detections of two runs on the same frames with the same rules. It shows how
similar the outputs are; it says nothing about which one is correct.

Values that cannot be computed (e.g. precision with no predictions) are
``None``, never 0. Importing this module needs NumPy only (``MissingDependencyError``
with an install hint without it).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from edgevision.optional import import_optional

np = import_optional("numpy", display_name="NumPy", extra="vision", purpose="compute detection metrics")

COCO_IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))  # 0.50 ... 0.95
RECALL_POINTS = np.linspace(0.0, 1.0, 101)


@dataclass(frozen=True)
class GroundTruthBox:
    """One labelled object: ``(x1, y1, x2, y2)`` in image pixels and the model's class ID."""

    bbox_xyxy: tuple[float, float, float, float]
    class_id: int
    crowd: bool = False  # a region with many objects: predictions inside it are neither TP nor FP


# ---------------------------------------------------------------------------
# Geometry and matching
# ---------------------------------------------------------------------------


def box_iou(a: Any, b: Any) -> np.ndarray:
    """IoU matrix ``(len(a), len(b))`` for boxes given as ``x1, y1, x2, y2``. Zero-area unions give 0."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if np.any(a[:, 2:] < a[:, :2]) or np.any(b[:, 2:] < b[:, :2]):
        raise ValueError("boxes must be x1, y1, x2, y2 with x1 <= x2 and y1 <= y2")
    inter_w = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    inter_h = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = inter_w * inter_h
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


TP, FP, IGNORED = 1, 0, -1


def _intersection_over_first(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Share of each box in ``a`` covered by each box in ``b`` (COCO's rule for crowd regions)."""
    inter_w = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    inter_h = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    inter = inter_w * inter_h
    return np.divide(inter, area_a, out=np.zeros_like(inter), where=area_a > 0)


def match_image(
    pred_boxes: Any, pred_scores: Any, pred_classes: Any,
    gt_boxes: Any, gt_classes: Any, gt_crowd: Any, iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Match one image's predictions to its ground truth.

    Returns ``(outcome, matched_gt)`` in the original prediction order:
    outcome is TP (1), FP (0) or IGNORED (-1); matched_gt is the index of the
    matched ground-truth box or -1.
    """
    pred_boxes = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
    pred_scores = np.asarray(pred_scores, dtype=np.float64).reshape(-1)
    pred_classes = np.asarray(pred_classes).reshape(-1)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    gt_classes = np.asarray(gt_classes).reshape(-1)
    gt_crowd = np.asarray(gt_crowd, dtype=bool).reshape(-1)
    if not len(pred_boxes) == len(pred_scores) == len(pred_classes):
        raise ValueError("predictions need the same number of boxes, scores and classes")
    if not len(gt_boxes) == len(gt_classes) == len(gt_crowd):
        raise ValueError("ground truth needs the same number of boxes, classes and crowd flags")

    outcome = np.full(len(pred_boxes), FP, dtype=np.int8)
    matched_gt = np.full(len(pred_boxes), -1, dtype=np.int64)
    if not len(pred_boxes):
        return outcome, matched_gt
    ious = box_iou(pred_boxes, gt_boxes) if len(gt_boxes) else np.zeros((len(pred_boxes), 0))
    inside = _intersection_over_first(pred_boxes, gt_boxes) if gt_crowd.any() else ious  # COCO: crowd uses IoA
    taken = np.zeros(len(gt_boxes), dtype=bool)
    for p in np.argsort(-pred_scores, kind="stable"):
        same_class = gt_classes == pred_classes[p]
        candidates = same_class & ~gt_crowd & ~taken & (ious[p] >= iou_threshold)
        if candidates.any():
            g = int(np.flatnonzero(candidates)[np.argmax(ious[p][candidates])])  # best IoU, lowest index on ties
            taken[g] = True
            outcome[p], matched_gt[p] = TP, g
        elif (same_class & gt_crowd & (inside[p] >= iou_threshold)).any():
            outcome[p] = IGNORED
    return outcome, matched_gt


def average_precision(scores: Any, outcomes: Any, num_ground_truth: int) -> float | None:
    """COCO 101-point interpolated AP for one class. ``None`` if the class has no ground truth.

    ``scores`` and ``outcomes`` (TP/FP/IGNORED) are for all predictions of the
    class across all images; ignored predictions are left out.
    """
    if num_ground_truth <= 0:
        return None
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    outcomes = np.asarray(outcomes).reshape(-1)
    keep = outcomes != IGNORED
    scores, outcomes = scores[keep], outcomes[keep]
    if not len(scores):
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    tp = np.cumsum(outcomes[order] == TP)
    fp = np.cumsum(outcomes[order] == FP)
    recall = tp / num_ground_truth
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    for i in range(len(precision) - 2, -1, -1):  # make precision non-increasing
        precision[i] = max(precision[i], precision[i + 1])
    indices = np.searchsorted(recall, RECALL_POINTS, side="left")
    sampled = np.array([precision[i] if i < len(precision) else 0.0 for i in indices])
    return float(sampled.mean())


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Evaluation against ground truth
# ---------------------------------------------------------------------------


def evaluate_detections(
    ground_truth: Mapping[Any, Sequence[GroundTruthBox]],
    predictions: Mapping[Any, Sequence[Any]],
    class_ids: Sequence[int],
    *,
    iou_thresholds: Sequence[float] = COCO_IOU_THRESHOLDS,
    pr_iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Precision, recall, F1 (at ``pr_iou_threshold``) and AP / mAP for ``class_ids``.

    ``ground_truth`` and ``predictions`` are keyed by the same image keys;
    every labelled image needs a predictions entry (an empty list if nothing
    was detected). Predictions are objects with ``bbox_xyxy``, ``confidence``
    and ``class_id`` (e.g. ``Detection``). Predictions of classes outside
    ``class_ids`` are not evaluated and are counted separately.
    """
    missing = [k for k in ground_truth if k not in predictions]
    extra = [k for k in predictions if k not in ground_truth]
    if missing or extra:
        raise ValueError(
            f"predictions and ground truth must cover the same images (missing predictions for {len(missing)}, "
            f"predictions for {len(extra)} unlabelled image(s))"
        )
    thresholds = sorted({float(t) for t in iou_thresholds} | {float(pr_iou_threshold)})
    if not all(0 < t <= 1 for t in thresholds):
        raise ValueError(f"IoU thresholds must be in (0, 1], got {thresholds}")
    classes = sorted({int(c) for c in class_ids})

    num_gt = {c: 0 for c in classes}
    num_pred = {c: 0 for c in classes}
    scores: dict[int, list[float]] = {c: [] for c in classes}
    outcomes: dict[tuple[int, float], list[int]] = {(c, t): [] for c in classes for t in thresholds}
    outside = 0

    for key, gts in ground_truth.items():
        preds = [p for p in predictions[key]]
        outside += sum(1 for p in preds if int(p.class_id) not in num_gt)
        preds = [p for p in preds if int(p.class_id) in num_gt]
        gts = [g for g in gts if int(g.class_id) in num_gt]
        for g in gts:
            if not g.crowd:
                num_gt[int(g.class_id)] += 1
        p_boxes = [p.bbox_xyxy for p in preds]
        p_scores = [float(p.confidence) for p in preds]
        p_classes = [int(p.class_id) for p in preds]
        for c, s in zip(p_classes, p_scores):
            num_pred[c] += 1
            scores[c].append(s)
        g_boxes = [g.bbox_xyxy for g in gts]
        g_classes = [int(g.class_id) for g in gts]
        g_crowd = [bool(g.crowd) for g in gts]
        for t in thresholds:
            result, _ = match_image(p_boxes, p_scores, p_classes, g_boxes, g_classes, g_crowd, t)
            for c, o in zip(p_classes, result.tolist()):
                outcomes[(c, t)].append(o)

    coco = [float(t) for t in iou_thresholds]
    per_class: dict[int, dict[str, Any]] = {}
    totals = {"tp": 0, "fp": 0, "fn": 0, "ignored": 0}
    for c in classes:
        at_pr = np.asarray(outcomes[(c, float(pr_iou_threshold))], dtype=np.int8)
        tp, fp, ignored = int((at_pr == TP).sum()), int((at_pr == FP).sum()), int((at_pr == IGNORED).sum())
        fn = num_gt[c] - tp
        precision, recall = _ratio(tp, tp + fp), _ratio(tp, num_gt[c])
        aps = [average_precision(scores[c], outcomes[(c, t)], num_gt[c]) for t in coco]
        ap50 = average_precision(scores[c], outcomes[(c, 0.5)], num_gt[c]) if 0.5 in thresholds else None
        per_class[c] = {
            "ground_truth": num_gt[c], "predictions": num_pred[c], "tp": tp, "fp": fp, "fn": fn, "ignored": ignored,
            "precision": precision, "recall": recall, "f1": _f1(precision, recall),
            "ap50": ap50, "ap50_95": None if aps[0] is None else float(np.mean(aps)),
        }
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals["ignored"] += ignored

    evaluated = [c for c in classes if num_gt[c] > 0]
    precision = _ratio(totals["tp"], totals["tp"] + totals["fp"])
    recall = _ratio(totals["tp"], totals["tp"] + totals["fn"])
    return {
        "images": len(ground_truth),
        "pr_iou_threshold": float(pr_iou_threshold),
        "map_iou_thresholds": coco,
        "per_class": per_class,
        "overall": {
            **totals,
            "precision": precision, "recall": recall, "f1": _f1(precision, recall),
            "map50": float(np.mean([per_class[c]["ap50"] for c in evaluated])) if evaluated and 0.5 in thresholds else None,
            "map50_95": float(np.mean([per_class[c]["ap50_95"] for c in evaluated])) if evaluated else None,
            "classes_with_ground_truth": len(evaluated),
        },
        "classes_without_ground_truth": [c for c in classes if num_gt[c] == 0],
        "predictions_outside_evaluated_classes": outside,
    }


# ---------------------------------------------------------------------------
# Agreement between two sets of detections (no ground truth)
# ---------------------------------------------------------------------------


def agreement(
    frames_a: Sequence[Sequence[Any]], frames_b: Sequence[Sequence[Any]], *, iou_threshold: float = 0.5
) -> dict[str, Any]:
    """How closely two detectors' outputs on the same frames agree. Not an accuracy measure.

    Frame by frame, B's detections (highest confidence first) are matched to
    A's detections of the same class with IoU >= ``iou_threshold``, each A
    detection at most once. Reports matched pairs, detections only one side
    produced, and how close matched boxes and confidences are.
    """
    if len(frames_a) != len(frames_b):
        raise ValueError(f"both sides need the same frames, got {len(frames_a)} and {len(frames_b)}")
    matched = only_a = only_b = 0
    ious: list[float] = []
    conf_diffs: list[float] = []
    per_class: dict[str, dict[str, int]] = {}

    def bump(name: str, field: str) -> None:
        per_class.setdefault(name, {"matched": 0, "only_a": 0, "only_b": 0})[field] += 1

    frames_identical = 0
    for dets_a, dets_b in zip(frames_a, frames_b):
        a_boxes = [d.bbox_xyxy for d in dets_a]
        outcome, match = match_image(
            [d.bbox_xyxy for d in dets_b], [d.confidence for d in dets_b], [d.class_id for d in dets_b],
            a_boxes, [d.class_id for d in dets_a], [False] * len(dets_a), iou_threshold,
        )
        used = set()
        iou = box_iou([d.bbox_xyxy for d in dets_b], a_boxes) if dets_a and dets_b else None
        for i, (o, g) in enumerate(zip(outcome.tolist(), match.tolist())):
            if o == TP:
                matched += 1
                used.add(g)
                ious.append(float(iou[i, g]))
                conf_diffs.append(abs(float(dets_b[i].confidence) - float(dets_a[g].confidence)))
                bump(dets_a[g].class_name, "matched")
            else:
                only_b += 1
                bump(dets_b[i].class_name, "only_b")
        for g, d in enumerate(dets_a):
            if g not in used:
                only_a += 1
                bump(d.class_name, "only_a")
        if len(used) == len(dets_a) == len(dets_b):
            frames_identical += 1

    total = matched + only_a + only_b
    return {
        "frames": len(frames_a),
        "iou_threshold": iou_threshold,
        "detections_a": matched + only_a,
        "detections_b": matched + only_b,
        "matched": matched,
        "only_a": only_a,
        "only_b": only_b,
        "matched_share": _ratio(matched, total),
        "frames_with_all_detections_matched": frames_identical,
        "matched_iou": _summary(ious),
        "matched_confidence_abs_diff": _summary(conf_diffs),
        "per_class": dict(sorted(per_class.items())),
    }


def _summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.asarray(values)
    return {"count": len(values), "mean": float(arr.mean()), "median": float(np.median(arr)),
            "min": float(arr.min()), "max": float(arr.max())}
