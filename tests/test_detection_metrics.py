"""Tests for edgevision.evaluation.detection_metrics.

All inputs are synthetic and hand-computed; they test the metric code, not
any model. When pycocotools is installed, AP/mAP are also checked against the
official COCO evaluation on random synthetic datasets.
"""

from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

from edgevision.evaluation.detection_metrics import (  # noqa: E402
    FP,
    IGNORED,
    TP,
    GroundTruthBox,
    agreement,
    average_precision,
    box_iou,
    evaluate_detections,
    match_image,
)
from edgevision.models.base import Detection  # noqa: E402


def pred(box, conf, cls=0):
    return SimpleNamespace(bbox_xyxy=tuple(box), confidence=conf, class_id=cls)


# --- IoU -----------------------------------------------------------------------------------------


def test_box_iou_values() -> None:
    a = [[0, 0, 10, 10]]
    b = [[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30], [0, 0, 10, 5]]
    np.testing.assert_allclose(box_iou(a, b), [[1.0, 50 / 150, 0.0, 0.5]])
    assert box_iou([[5, 5, 5, 5]], [[5, 5, 5, 5]])[0, 0] == 0.0  # zero-area boxes: no division by zero
    assert box_iou(np.zeros((0, 4)), a).shape == (0, 1)


def test_box_iou_rejects_inverted_boxes() -> None:
    with pytest.raises(ValueError, match="x1 <= x2"):
        box_iou([[10, 0, 0, 10]], [[0, 0, 1, 1]])


# --- matching ------------------------------------------------------------------------------------


def match(preds, gts, iou=0.5):
    return match_image([p[0] for p in preds], [p[1] for p in preds], [p[2] for p in preds],
                       [g[0] for g in gts], [g[1] for g in gts], [g[2] if len(g) > 2 else False for g in gts], iou)


def test_highest_confidence_is_matched_first() -> None:
    gts = [([0, 0, 10, 10], 0)]
    outcome, matched = match([([0, 0, 10, 9], 0.6, 0), ([1, 1, 10, 10], 0.9, 0)], gts)
    assert outcome.tolist() == [FP, TP]  # both overlap, the more confident one gets the object
    assert matched.tolist() == [-1, 0]


def test_each_prediction_takes_the_best_unmatched_box() -> None:
    gts = [([0, 0, 10, 10], 0), ([2, 0, 12, 10], 0)]
    outcome, matched = match([([2, 0, 12, 10], 0.9, 0), ([0, 0, 10, 10], 0.8, 0)], gts)
    assert outcome.tolist() == [TP, TP] and matched.tolist() == [1, 0]


def test_class_must_match_and_threshold_is_inclusive() -> None:
    gts = [([0, 0, 10, 10], 0)]
    assert match([([0, 0, 10, 10], 0.9, 1)], gts)[0].tolist() == [FP]  # right place, wrong class
    assert match([([0, 0, 10, 5], 0.9, 0)], gts, iou=0.5)[0].tolist() == [TP]  # IoU exactly 0.5 counts
    assert match([([0, 0, 10, 5], 0.9, 0)], gts, iou=0.51)[0].tolist() == [FP]


def test_crowd_regions_are_ignored_and_empty_inputs() -> None:
    gts = [([0, 0, 100, 100], 0, True)]
    assert match([([10, 10, 90, 90], 0.9, 0)], gts)[0].tolist() == [IGNORED]
    assert match([], gts)[0].tolist() == []
    assert match([([0, 0, 1, 1], 0.5, 0)], [])[0].tolist() == [FP]


# --- average precision -------------------------------------------------------------------------


def test_average_precision_hand_computed() -> None:
    # Two objects; predictions TP (0.9), FP (0.8), TP (0.7):
    # recall 0.5, 0.5, 1.0; precision 1, 0.5, 0.667 -> made non-increasing: 1, 0.667, 0.667.
    # 51 of the 101 recall points (0 ... 0.5) see precision 1, the other 50 see 2/3.
    ap = average_precision([0.9, 0.8, 0.7], [TP, FP, TP], num_ground_truth=2)
    assert ap == pytest.approx((51 * 1 + 50 * 2 / 3) / 101)


@pytest.mark.parametrize(
    ("scores", "outcomes", "num_gt", "expected"),
    [
        ([0.9, 0.8], [TP, TP], 2, 1.0),  # everything found, nothing wrong
        ([0.9, 0.8], [FP, FP], 2, 0.0),
        ([], [], 3, 0.0),  # objects but no predictions: a real 0
        ([0.9], [TP], 0, None),  # no ground truth: undefined, not 0
        ([0.9, 0.8], [IGNORED, TP], 1, 1.0),  # ignored predictions do not count
    ],
)
def test_average_precision_edge_cases(scores, outcomes, num_gt, expected) -> None:
    assert average_precision(scores, outcomes, num_gt) == expected


# --- evaluation ----------------------------------------------------------------------------------


def test_evaluate_counts_and_ratios() -> None:
    gt = {
        "a": [GroundTruthBox((0, 0, 10, 10), 0), GroundTruthBox((50, 50, 60, 60), 1)],
        "b": [GroundTruthBox((0, 0, 20, 20), 0)],
    }
    preds = {
        "a": [pred((0, 0, 10, 10), 0.9, 0), pred((70, 70, 80, 80), 0.8, 0), pred((50, 50, 60, 60), 0.7, 5)],
        "b": [],
    }
    m = evaluate_detections(gt, preds, class_ids=[0, 1, 2])
    c0, c1, c2 = (m["per_class"][c] for c in (0, 1, 2))
    assert (c0["tp"], c0["fp"], c0["fn"]) == (1, 1, 1)
    assert c0["precision"] == 0.5 and c0["recall"] == 0.5 and c0["f1"] == 0.5
    assert (c1["tp"], c1["fp"], c1["fn"], c1["predictions"]) == (0, 0, 1, 0)
    assert c1["precision"] is None and c1["recall"] == 0.0 and c1["f1"] is None  # not 0: nothing was predicted
    assert c1["ap50"] == 0.0  # it had an object and found nothing
    assert c2["ap50"] is None and m["classes_without_ground_truth"] == [2]
    assert m["predictions_outside_evaluated_classes"] == 1  # the class-5 prediction
    o = m["overall"]
    assert (o["tp"], o["fp"], o["fn"]) == (1, 1, 2)
    assert o["precision"] == 0.5 and o["recall"] == pytest.approx(1 / 3)
    assert o["classes_with_ground_truth"] == 2
    assert o["map50"] == pytest.approx((c0["ap50"] + 0.0) / 2)


def test_map50_95_averages_iou_thresholds() -> None:
    gt = {1: [GroundTruthBox((0, 0, 10, 10), 0)]}
    m = evaluate_detections(gt, {1: [pred((0, 0, 10, 8), 0.9)]}, [0])  # IoU exactly 0.8
    assert m["per_class"][0]["ap50"] == 1.0
    assert m["per_class"][0]["ap50_95"] == pytest.approx(0.7)  # matched at 0.50 ... 0.80, missed at 0.85 ... 0.95
    perfect = evaluate_detections(gt, {1: [pred((0, 0, 10, 10), 0.9)]}, [0])
    assert perfect["overall"]["map50"] == perfect["overall"]["map50_95"] == 1.0


def test_crowd_boxes_are_not_objects_to_find() -> None:
    gt = {1: [GroundTruthBox((0, 0, 100, 100), 0, crowd=True)]}
    m = evaluate_detections(gt, {1: [pred((10, 10, 50, 50), 0.9)]}, [0])
    c = m["per_class"][0]
    assert (c["ground_truth"], c["tp"], c["fp"], c["ignored"]) == (0, 0, 0, 1)


def test_every_labelled_image_needs_predictions() -> None:
    gt = {1: [], 2: []}
    with pytest.raises(ValueError, match="missing predictions for 1"):
        evaluate_detections(gt, {1: []}, [0])
    with pytest.raises(ValueError, match="1 unlabelled"):
        evaluate_detections(gt, {1: [], 2: [], 3: []}, [0])
    with pytest.raises(ValueError, match="IoU thresholds"):
        evaluate_detections(gt, {1: [], 2: []}, [0], iou_thresholds=[0.0])


def test_matches_official_coco_evaluation() -> None:
    """Random synthetic datasets (with crowd regions): AP@0.5 and AP@0.5:0.95 must equal pycocotools' COCOeval."""
    coco_module = pytest.importorskip("pycocotools.coco", reason="pycocotools not installed (optional cross-check)")
    cocoeval = pytest.importorskip("pycocotools.cocoeval")
    for seed in range(3):
        rng = np.random.default_rng(seed)
        images, anns, dets, gt, preds = [], [], [], {}, {}
        for i in range(1, 31):
            images.append({"id": i, "file_name": f"{i}.jpg", "width": 640, "height": 480})
            gt[i], preds[i] = [], []
            for _ in range(rng.integers(0, 7)):
                c = int(rng.integers(0, 3))
                x, y, w, h = rng.uniform(0, 500), rng.uniform(0, 380), rng.uniform(8, 140), rng.uniform(8, 100)
                crowd = bool(rng.random() < 0.15)  # crowd region: detections inside it are ignored
                anns.append({"id": len(anns) + 1, "image_id": i, "category_id": c + 1, "bbox": [x, y, w, h],
                             "area": w * h, "iscrowd": int(crowd)})
                gt[i].append(GroundTruthBox((x, y, x + w, y + h), c, crowd))
                if crowd:
                    b = [x + w * 0.3, y + h * 0.3, w * 0.3, h * 0.3]
                    s = float(rng.uniform(0.05, 1))
                    dets.append({"image_id": i, "category_id": c + 1, "bbox": b, "score": s})
                    preds[i].append(pred((b[0], b[1], b[0] + b[2], b[1] + b[3]), s, c))
                elif rng.random() < 0.8:
                    j = rng.normal(0, 6, 4)
                    pc = c if rng.random() < 0.9 else int(rng.integers(0, 3))
                    b = [x + j[0], y + j[1], max(1, w + j[2]), max(1, h + j[3])]
                    s = float(rng.uniform(0.05, 1))
                    dets.append({"image_id": i, "category_id": pc + 1, "bbox": b, "score": s})
                    preds[i].append(pred((b[0], b[1], b[0] + b[2], b[1] + b[3]), s, pc))
            for _ in range(rng.integers(0, 3)):
                pc = int(rng.integers(0, 3))
                b = [rng.uniform(0, 500), rng.uniform(0, 380), rng.uniform(8, 140), rng.uniform(8, 100)]
                s = float(rng.uniform(0.05, 1))
                dets.append({"image_id": i, "category_id": pc + 1, "bbox": b, "score": s})
                preds[i].append(pred((b[0], b[1], b[0] + b[2], b[1] + b[3]), s, pc))
        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt = coco_module.COCO()
            coco_gt.dataset = {"images": images, "annotations": anns,
                               "categories": [{"id": c + 1, "name": f"c{c}"} for c in range(3)]}
            coco_gt.createIndex()
            ev = cocoeval.COCOeval(coco_gt, coco_gt.loadRes(dets), "bbox")
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        ours = evaluate_detections(gt, preds, [0, 1, 2])
        precision = ev.eval["precision"]  # [IoU, recall, class, area, maxDets]
        for c in range(3):
            per_iou = [precision[t, :, c, 0, 2].mean() for t in range(10)]
            assert ours["per_class"][c]["ap50"] == pytest.approx(per_iou[0], abs=1e-12)
            assert ours["per_class"][c]["ap50_95"] == pytest.approx(np.mean(per_iou), abs=1e-12)
        assert ours["overall"]["map50"] == pytest.approx(ev.stats[1], abs=1e-12)
        assert ours["overall"]["map50_95"] == pytest.approx(ev.stats[0], abs=1e-12)


# --- agreement between two runs ------------------------------------------------------------------


def det(frame, box, conf, cls=0, name="person"):
    return Detection(frame, box, conf, cls, name)


def test_identical_outputs_agree_fully() -> None:
    frames = [[det(0, (0, 0, 10, 10), 0.9)], [], [det(2, (5, 5, 20, 20), 0.6, 15, "cat")]]
    r = agreement(frames, frames)
    assert (r["matched"], r["only_a"], r["only_b"], r["matched_share"]) == (2, 0, 0, 1.0)
    assert r["frames_with_all_detections_matched"] == 3
    assert r["matched_iou"]["min"] == 1.0 and r["matched_confidence_abs_diff"]["max"] == 0.0


def test_differences_are_counted_per_side_and_class() -> None:
    a = [[det(0, (0, 0, 10, 10), 0.9), det(0, (50, 50, 60, 60), 0.4, 15, "cat")]]
    b = [[det(0, (0, 0, 10, 9), 0.8), det(0, (50, 50, 60, 60), 0.5, 16, "dog")]]
    r = agreement(a, b)
    assert (r["matched"], r["only_a"], r["only_b"]) == (1, 1, 1)  # same place, different class: no match
    assert r["matched_share"] == pytest.approx(1 / 3)
    assert r["per_class"] == {"cat": {"matched": 0, "only_a": 1, "only_b": 0},
                              "dog": {"matched": 0, "only_a": 0, "only_b": 1},
                              "person": {"matched": 1, "only_a": 0, "only_b": 0}}
    assert r["matched_iou"]["mean"] == pytest.approx(0.9)
    assert r["matched_confidence_abs_diff"]["mean"] == pytest.approx(0.1)


def test_agreement_needs_the_same_frames() -> None:
    with pytest.raises(ValueError, match="same frames"):
        agreement([[]], [[], []])
    empty = agreement([[]], [[]])
    assert empty["matched_share"] is None and empty["matched_iou"] is None
