"""Tests for edgevision.models.onnx_backend.

* Pre/postprocessing functions are tested directly on synthetic arrays
  (NumPy; OpenCV for resizing).
* ``OnnxYoloDetector`` is tested with the fake ONNX Runtime from conftest.py
  (``fake_ort``): no ONNX Runtime, no model file contents, no GPU needed.
* When Ultralytics/PyTorch are installed, the letterbox and NMS code is also
  compared with Ultralytics' own implementations on random inputs.
* An opt-in test runs a real exported model (marker ``model``).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from edgevision.models.base import (  # noqa: E402
    DetectionSettings,
    DetectorError,
    DeviceUnavailableError,
    InferenceError,
    ModelLoadError,
)
from edgevision.models.onnx_backend import (  # noqa: E402
    MAX_DETECTIONS,
    OUTPUT_NMS,
    OUTPUT_RAW,
    Letterbox,
    OnnxYoloDetector,
    decode_nms_output,
    decode_raw_output,
    nms,
    restore_boxes,
    select_providers,
)
from edgevision.models import onnx_backend  # noqa: E402
from edgevision.optional import MissingDependencyError  # noqa: E402

CPU, CUDA = "CPUExecutionProvider", "CUDAExecutionProvider"
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def cv2():
    return pytest.importorskip("cv2", reason='OpenCV not installed (pip install -e ".[vision]")')


@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"not read by the fake session")
    return path


def settings(**kw) -> DetectionSettings:
    return DetectionSettings(**kw)


# --- execution providers -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("device", "available", "expected"),
    [
        ("cpu", [CPU], [CPU]),
        ("cpu", [CUDA, CPU], [CPU]),  # an explicit CPU request never uses the GPU
        ("auto", [CPU], [CPU]),
        ("auto", ["AzureExecutionProvider", CPU], [CPU]),
        ("auto", [CUDA, CPU], [CUDA, CPU]),
        ("cuda", [CUDA, CPU], [CUDA, CPU]),
    ],
)
def test_select_providers(device: str, available: list[str], expected: list[str]) -> None:
    assert select_providers(device, available) == expected


def test_cuda_without_cuda_provider_is_an_error() -> None:
    with pytest.raises(DeviceUnavailableError, match=r"no CUDAExecutionProvider.*CPU-only.*onnxruntime-gpu"):
        select_providers("cuda", [CPU])


@pytest.mark.parametrize("device", ["gpu", "CUDA", None, 0])
def test_invalid_device(device) -> None:
    with pytest.raises(ValueError, match="device must be one of"):
        select_providers(device, [CPU])


def test_no_cpu_provider() -> None:
    with pytest.raises(DeviceUnavailableError, match="no CPUExecutionProvider"):
        select_providers("auto", ["SomethingElse"])


# --- preprocessing --------------------------------------------------------------------------


def test_preprocess_shape_dtype_and_range(cv2) -> None:
    from edgevision.models.onnx_backend import preprocess

    frame = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    tensor, box = preprocess(frame, 640, cv2)
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32 and tensor.flags["C_CONTIGUOUS"]
    assert 0.0 <= tensor.min() and tensor.max() <= 1.0
    assert box == Letterbox(1.0, 1.0, 0, 80)  # 480 rows centred: 80 rows of padding above and below
    # Padding is grey 114, the image itself is unchanged apart from scaling to 0-1.
    np.testing.assert_allclose(tensor[0, :, :80, :], 114 / 255, rtol=1e-6)
    np.testing.assert_allclose(tensor[0, :, 560:, :], 114 / 255, rtol=1e-6)
    np.testing.assert_allclose(tensor[0, :, 80:560, :], frame[:, :, ::-1].transpose(2, 0, 1) / 255, rtol=1e-6)


def test_preprocess_converts_bgr_to_rgb(cv2) -> None:
    from edgevision.models.onnx_backend import preprocess

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[:, :] = (255, 0, 0)  # pure blue in OpenCV's BGR order
    tensor, _ = preprocess(frame, 64, cv2)
    assert tensor[0, 0].max() == 0.0  # R
    assert tensor[0, 1].max() == 0.0  # G
    assert tensor[0, 2].min() == 1.0  # B


@pytest.mark.parametrize(
    ("height", "width", "size", "expected"),
    [
        (1080, 810, 640, Letterbox(480 / 810, 640 / 1080, 80, 0)),  # portrait: padding left and right
        (576, 1024, 640, Letterbox(640 / 1024, 360 / 576, 0, 140)),  # landscape: padding above and below
        (51, 100, 64, Letterbox(64 / 100, 33 / 51, 0, 15)),  # odd padding: 15 above, 16 below
        (32, 32, 64, Letterbox(2.0, 2.0, 0, 0)),  # small images are scaled up
    ],
)
def test_letterbox_geometry(cv2, height: int, width: int, size: int, expected: Letterbox) -> None:
    from edgevision.models.onnx_backend import letterbox

    image, box = letterbox(np.zeros((height, width, 3), np.uint8), size, cv2)
    assert image.shape == (size, size, 3)
    assert box.pad_x == expected.pad_x and box.pad_y == expected.pad_y
    assert box.gain_x == pytest.approx(expected.gain_x) and box.gain_y == pytest.approx(expected.gain_y)


@pytest.mark.parametrize(("height", "width"), [(1080, 810), (576, 1024), (51, 100), (480, 640), (17, 333)])
def test_letterbox_matches_ultralytics(cv2, height: int, width: int) -> None:
    augment = pytest.importorskip("ultralytics.data.augment", reason="Ultralytics not installed")
    from edgevision.models.onnx_backend import letterbox

    frame = np.random.default_rng(height * width).integers(0, 256, (height, width, 3), dtype=np.uint8)
    ours, _ = letterbox(frame, 640, cv2)
    theirs = augment.LetterBox(640, auto=False, stride=32)(image=frame)
    np.testing.assert_array_equal(ours, theirs)  # pixel-identical


# --- postprocessing -------------------------------------------------------------------------


def test_nms_keeps_the_best_of_overlapping_boxes() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    scores = np.array([0.8, 0.9, 0.7], np.float32)
    assert nms(boxes, scores, 0.5).tolist() == [1, 2]  # box 0 overlaps box 1 (IoU 0.68) and loses


def test_nms_threshold_is_exclusive() -> None:
    # IoU of these two boxes is exactly 0.5: kept at threshold 0.5, suppressed below it.
    boxes = np.array([[0, 0, 2, 1], [0, 0, 1, 1]], np.float32)
    scores = np.array([0.9, 0.8], np.float32)
    assert nms(boxes, scores, 0.5).tolist() == [0, 1]
    assert nms(boxes, scores, 0.49).tolist() == [0]


def test_nms_handles_empty_and_degenerate_boxes() -> None:
    assert nms(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), 0.5).tolist() == []
    flat = np.array([[5, 5, 5, 5], [5, 5, 5, 5]], np.float32)  # zero area: IoU 0, no division by zero
    assert nms(flat, np.array([0.9, 0.8], np.float32), 0.5).tolist() == [0, 1]


def test_decode_raw_output(make_raw_output) -> None:
    out = make_raw_output([
        (100, 100, 40, 20, 2, 0.90),  # cat
        (102, 101, 40, 20, 2, 0.80),  # same cat, overlapping: removed by NMS
        (102, 101, 40, 20, 0, 0.85),  # person at the same place: other class, kept
        (300, 300, 10, 10, 1, 0.25),  # exactly the threshold: dropped (score must be greater)
        (400, 400, 10, 10, 1, 0.26),  # bus just above the threshold
    ])[0]
    boxes, scores, class_ids = decode_raw_output(out, 3, settings(conf_threshold=0.25, iou_threshold=0.7))
    assert class_ids.tolist() == [2, 0, 1]
    np.testing.assert_allclose(scores, [0.90, 0.85, 0.26], rtol=1e-6)
    np.testing.assert_allclose(boxes[0], [80, 90, 120, 110])  # centre/size -> corners
    np.testing.assert_allclose(boxes[2], [395, 395, 405, 405])


def test_decode_raw_output_uses_the_best_class_per_candidate(make_raw_output) -> None:
    out = make_raw_output([(50, 50, 10, 10, 0, 0.4)])
    out[0, 4 + 2, 0] = 0.7  # the same candidate scores higher as a cat
    _, scores, class_ids = decode_raw_output(out[0], 3, settings())
    assert class_ids.tolist() == [2] and scores.tolist() == pytest.approx([0.7])


def test_decode_raw_output_class_filter(make_raw_output) -> None:
    out = make_raw_output([(50, 50, 10, 10, 0, 0.9), (150, 50, 10, 10, 1, 0.8), (250, 50, 10, 10, 2, 0.7)])[0]
    _, _, class_ids = decode_raw_output(out, 3, settings(classes=[0, 2]))
    assert class_ids.tolist() == [0, 2]


def test_decode_raw_output_limits_detections(make_raw_output) -> None:
    rows = [(10 + 20 * (i % 30), 10 + 20 * (i // 30), 5, 5, 0, 0.5 + i / 10_000) for i in range(400)]
    boxes, scores, _ = decode_raw_output(make_raw_output(rows)[0], 3, settings())
    assert len(boxes) == MAX_DETECTIONS
    assert np.all(np.diff(scores) <= 0)  # highest score first


def test_decode_raw_output_empty(make_raw_output) -> None:
    boxes, scores, class_ids = decode_raw_output(make_raw_output([])[0], 3, settings())
    assert boxes.shape == (0, 4) and scores.shape == (0,) and class_ids.shape == (0,)


def test_decode_raw_output_rejects_wrong_class_count(make_raw_output) -> None:
    with pytest.raises(InferenceError, match="4 box values \\+ 80 class scores"):
        decode_raw_output(make_raw_output([])[0], 80, settings())


def _transposed_reference(output, settings):
    """The straightforward per-candidate decoding (transpose, best class, threshold, class filter)."""
    preds = output.T.astype(np.float32)
    ids = preds[:, 4:].argmax(axis=1)
    scores = preds[np.arange(len(preds)), 4 + ids]
    keep = scores > settings.conf_threshold
    if settings.classes is not None:
        keep &= np.isin(ids, settings.classes)
    xywh = preds[keep, :4]
    boxes = np.concatenate([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], axis=1)
    offsets = (ids[keep].astype(np.float32) * np.float32(onnx_backend.CLASS_OFFSET))[:, None]
    kept = onnx_backend.nms(boxes + offsets, scores[keep], settings.iou_threshold)[:onnx_backend.MAX_DETECTIONS]
    return boxes[kept], scores[keep][kept], ids[keep][kept]


@pytest.mark.parametrize("classes", [None, (0, 2, 5)])
def test_decode_raw_output_equals_transposed_reference(classes) -> None:
    # The decoder filters candidates on the model's (classes, candidates) layout for speed; the result must be
    # bit-identical to decoding each transposed candidate row, including ties between class scores (first class
    # wins), float16 outputs and scores exactly at the threshold (dropped).
    rng = np.random.default_rng(7)
    settings = DetectionSettings(conf_threshold=0.25, iou_threshold=0.7, classes=classes)
    for dtype in (np.float32, np.float16):
        output = np.zeros((4 + 8, 3000), dtype=np.float32)
        output[:2] = rng.uniform(0, 640, (2, 3000))
        output[2:4] = rng.uniform(2, 120, (2, 3000))
        output[4:] = rng.uniform(0, 0.5, (8, 3000)).round(2)  # rounding creates many tied class scores
        output[4:, :50] = 0.25  # exactly at the threshold
        output = output.astype(dtype)
        got, want = onnx_backend.decode_raw_output(output, 8, settings), _transposed_reference(output, settings)
        assert len(want[0]) > 20
        for g, w in zip(got, want):
            assert g.dtype == w.dtype or g.dtype.kind == w.dtype.kind == "i"
            np.testing.assert_array_equal(g, w)


def test_decode_nms_output_does_not_run_nms_again() -> None:
    rows = np.array([
        [10, 10, 50, 50, 0.6, 2],
        [11, 11, 51, 51, 0.9, 2],  # overlaps the first: both kept, the model already decided
        [0, 0, 0, 0, 0.0, 0],  # padding row
        [5, 5, 9, 9, 0.3, 1],
    ], np.float32)
    boxes, scores, class_ids = decode_nms_output(rows, settings(classes=[1, 2]))
    assert scores.tolist() == pytest.approx([0.9, 0.6, 0.3]) and class_ids.tolist() == [2, 2, 1]
    assert decode_nms_output(rows, settings(classes=[2]))[2].tolist() == [2, 2]


def test_decode_nms_output_rejects_wrong_shape() -> None:
    with pytest.raises(InferenceError, match=r"\(detections, 6\)"):
        decode_nms_output(np.zeros((300, 7), np.float32), settings())


def test_restore_boxes_undoes_letterbox_and_clips() -> None:
    box = Letterbox(480 / 810, 640 / 1080, 80, 0)  # 810 x 1080 portrait frame in a 640 input
    restored = restore_boxes(np.array([[80, 0, 560, 640], [100, 100, 200, 200], [0, -10, 700, 700]], np.float32),
                             box, 810, 1080)
    np.testing.assert_allclose(restored[0], [0, 0, 810, 1080], atol=1e-3)  # whole image area
    np.testing.assert_allclose(restored[1], [33.75, 168.75, 202.5, 337.5], atol=1e-3)
    np.testing.assert_allclose(restored[2], [0, 0, 810, 1080], atol=1e-3)  # clipped to the frame


def test_round_trip_from_original_pixels(cv2) -> None:
    """A box placed in original-frame pixels comes back at the same place."""
    from edgevision.models.onnx_backend import letterbox

    frame = np.zeros((576, 1024, 3), np.uint8)
    _, box = letterbox(frame, 640, cv2)
    original = np.array([[100, 50, 300, 400]], np.float32)
    in_input = original.copy()
    in_input[:, [0, 2]] = original[:, [0, 2]] * box.gain_x + box.pad_x
    in_input[:, [1, 3]] = original[:, [1, 3]] * box.gain_y + box.pad_y
    np.testing.assert_allclose(restore_boxes(in_input, box, 1024, 576), original, atol=1e-3)


@pytest.mark.parametrize(("conf", "iou", "classes"), [(0.25, 0.7, None), (0.1, 0.45, None), (0.3, 0.5, [0, 5, 7])])
def test_raw_decoding_matches_ultralytics_nms(conf: float, iou: float, classes) -> None:
    torch = pytest.importorskip("torch", reason="PyTorch not installed")
    nms_module = pytest.importorskip("ultralytics.utils.nms", reason="Ultralytics not installed")

    rng = np.random.default_rng(42)
    n, nc = 8400, 80
    out = np.zeros((1, 4 + nc, n), np.float32)
    out[0, 0:2] = rng.uniform(0, 640, (2, n))  # centres
    out[0, 2:4] = rng.uniform(4, 200, (2, n))  # sizes
    out[0, 4:] = rng.uniform(0, 1, (nc, n)) ** 12  # mostly low scores, some high: realistic sparsity
    ours = decode_raw_output(out[0], nc, settings(conf_threshold=conf, iou_threshold=iou, classes=classes))
    theirs = nms_module.non_max_suppression(torch.from_numpy(out), conf, iou, classes=classes, max_det=300)[0].numpy()
    assert len(ours[0]) == len(theirs) > 0
    # Same detections. The order may differ only between equal scores: PyTorch's
    # sort does not define an order for ties, so compare sorted rows.
    np.testing.assert_allclose(ours[1], theirs[:, 4], rtol=1e-6)  # scores, highest first, identical
    ours_rows = np.column_stack([ours[0], ours[1], ours[2]])
    theirs_rows = theirs[:, :6]
    order = lambda rows: rows[np.lexsort(rows.T[::-1])]  # noqa: E731
    np.testing.assert_allclose(order(ours_rows), order(theirs_rows), atol=1e-3)


# --- detector with a fake ONNX Runtime ---------------------------------------------------------


def test_missing_onnxruntime(model_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(MissingDependencyError) as info:
        OnnxYoloDetector(model_file)
    message = str(info.value)
    assert ".[onnxruntime]" in message and ".[onnxruntime-gpu]" in message and "only one" in message


def test_missing_opencv(fake_ort, model_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(MissingDependencyError, match=r"OpenCV.*\.\[vision\]"):
        OnnxYoloDetector(model_file)


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("missing.onnx", None, "ONNX model not found.*export_onnx.py.*never downloaded"),
        ("model.pt", b"x", "Expected an .onnx file"),
        ("empty.onnx", b"", "empty"),
    ],
)
def test_invalid_model_paths(fake_ort, tmp_path: Path, name: str, content, message: str) -> None:
    path = tmp_path / name
    if content is not None:
        path.write_bytes(content)
    with pytest.raises(ModelLoadError, match=message):
        OnnxYoloDetector(path)
    assert fake_ort.sessions == []  # ONNX Runtime was never asked to load it


def test_directory_is_not_a_model(fake_ort, tmp_path: Path) -> None:
    folder = tmp_path / "x.onnx"
    folder.mkdir()
    with pytest.raises(ModelLoadError, match="not a file"):
        OnnxYoloDetector(folder)


@pytest.mark.parametrize("model", ["", "   ", None, 3])
def test_invalid_model_argument(model) -> None:
    with pytest.raises(ValueError, match="model must be"):
        OnnxYoloDetector(model)


def test_invalid_settings_and_device(model_file: Path) -> None:
    with pytest.raises(TypeError, match="DetectionSettings"):
        OnnxYoloDetector(model_file, settings={"conf": 0.5})
    with pytest.raises(ValueError, match="device must be one of"):
        OnnxYoloDetector(model_file, device="gpu")


def test_session_load_failure(fake_ort, cv2, model_file: Path) -> None:
    fake_ort.load_error = RuntimeError("[ONNXRuntimeError] : 7 : INVALID_PROTOBUF : Load model failed")
    with pytest.raises(ModelLoadError, match="INVALID_PROTOBUF.*inspect_onnx.py") as info:
        OnnxYoloDetector(model_file)
    assert isinstance(info.value.__cause__, RuntimeError)


def test_cpu_session(fake_ort, cv2, model_file: Path) -> None:
    with OnnxYoloDetector(model_file, device="cpu") as detector:
        assert fake_ort.sessions == [{"path": str(model_file), "providers": [CPU]}]
        assert detector.providers == (CPU,) and detector.device == "cpu"
        assert detector.output_format == OUTPUT_RAW and detector.class_names == {0: "person", 1: "bus", 2: "cat"}
        assert detector.description == "ONNX Runtime 1.31.0, providers: CPUExecutionProvider"
        assert detector.backend_name == "onnx"
    assert detector.closed


def test_auto_uses_cuda_when_it_is_active(fake_ort, cv2, model_file: Path) -> None:
    fake_ort.available = [CUDA, CPU]
    detector = OnnxYoloDetector(model_file, device="auto")
    assert fake_ort.sessions[0]["providers"] == [CUDA, CPU]
    assert detector.providers == (CUDA, CPU) and detector.device == "cuda:0"


def test_auto_reports_cpu_when_cuda_cannot_start(fake_ort, cv2, model_file: Path, caplog) -> None:
    fake_ort.available = [CUDA, CPU]
    fake_ort.activated = [CPU]  # what ONNX Runtime does when CUDA/cuDNN libraries are missing
    with caplog.at_level(logging.WARNING):
        detector = OnnxYoloDetector(model_file, device="auto")
    assert detector.device == "cpu" and detector.providers == (CPU,)  # never claims the GPU
    assert "could not be activated" in caplog.text


def test_explicit_cuda_never_silently_runs_on_cpu(fake_ort, cv2, model_file: Path) -> None:
    fake_ort.available = [CUDA, CPU]
    fake_ort.activated = [CPU]
    with pytest.raises(DeviceUnavailableError, match="could not activate it.*cuDNN"):
        OnnxYoloDetector(model_file, device="cuda")


def test_explicit_cuda_without_gpu_build(fake_ort, cv2, model_file: Path) -> None:
    with pytest.raises(DeviceUnavailableError, match="no CUDAExecutionProvider"):
        OnnxYoloDetector(model_file, device="cuda")
    assert fake_ort.sessions == []  # failed before loading the model


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"input_type": "tensor(float16)"}, "only float32 inputs"),
        ({"input_shape": [1, 3, 640, 480]}, "only square inputs"),
        ({"input_shape": [4, 3, 640, 640]}, "fixed batch size of 4"),
        ({"input_shape": [1, 1, 640, 640]}, r"expected \(batch, 3, height, width\)"),
        ({"input_shape": [1, 3, 640]}, r"expected \(batch, 3, height, width\)"),
        ({"outputs": [("a", [1, 7, 10], "tensor(float)"), ("b", [1, 32, 8, 8], "tensor(float)")]}, "1 input and 1 output"),
        ({"outputs": [("output0", [1, 85, 8400], "tensor(float)")]}, r"matches neither .*4 \+ 3 classes"),
        ({"outputs": [("output0", [1, 8400], "tensor(float)")]}, "3-D float tensor"),
        ({"outputs": [("output0", [1, 7, 8400], "tensor(int64)")]}, "3-D float tensor"),
        ({"metadata": {"task": "segment", "names": "{0: 'a'}"}}, "task 'segment'"),
        ({"metadata": {"task": "detect"}}, "no readable class names"),
        ({"metadata": {"names": "__import__('os').getcwd()"}}, "no readable class names"),
        ({"metadata": {"names": "{0: 'a'}", "args": "{'nms': True}"}, "outputs": [("o", [1, 5, 8400], "tensor(float)")]},
         "NMS built in"),
    ],
)
def test_unsupported_models_are_rejected(fake_ort, cv2, model_file: Path, change: dict, message: str) -> None:
    for key, value in change.items():
        setattr(fake_ort, key, value)
    with pytest.raises(ModelLoadError, match=message) as info:
        OnnxYoloDetector(model_file)
    assert "inspect_onnx.py" in str(info.value)


def test_input_size_must_match_a_fixed_model(fake_ort, cv2, model_file: Path) -> None:
    fake_ort.input_shape = [1, 3, 320, 320]
    with pytest.raises(ModelLoadError, match="fixed 320 x 320 input, but input size 640.*--imgsz 320"):
        OnnxYoloDetector(model_file)
    assert OnnxYoloDetector(model_file, settings=settings(input_size=320)).layout.input_size == 320


def test_dynamic_model_uses_the_requested_size(fake_ort, cv2, model_file: Path) -> None:
    fake_ort.input_shape = ["batch", 3, "height", "width"]
    fake_ort.outputs = [("output0", ["batch", 7, "anchors"], "tensor(float)")]
    detector = OnnxYoloDetector(model_file, settings=settings(input_size=320))
    detector.detect(np.zeros((100, 200, 3), np.uint8))
    assert fake_ort.feeds[0]["images"].shape == (1, 3, 320, 320)


@pytest.mark.parametrize(
    ("metadata", "output_shape"),
    [
        ({"args": "{'nms': True}"}, [1, 300, 6]),  # Ultralytics export with nms=True
        ({"end2end": "True"}, [1, 300, 6]),  # end-to-end (NMS-free) models
        ({}, [1, 300, 6]),  # no metadata flag, but the shape is unambiguous
    ],
)
def test_nms_output_format_is_detected(fake_ort, cv2, model_file: Path, metadata: dict, output_shape: list) -> None:
    fake_ort.metadata = {"task": "detect", "names": "{0: 'person', 1: 'bus', 2: 'cat'}", **metadata}
    fake_ort.outputs = [("output0", output_shape, "tensor(float)")]
    assert OnnxYoloDetector(model_file).output_format == OUTPUT_NMS


def test_detect_end_to_end(fake_ort, cv2, model_file: Path, make_raw_output) -> None:
    # 640 x 480 frame -> 640 x 640 input with 80 rows of padding above and below.
    fake_ort.output = make_raw_output([
        (320, 320, 100, 60, 2, 0.9),  # a cat in the middle of the input
        (100, 100, 20, 20, 0, 0.1),  # below the confidence threshold
    ])
    with OnnxYoloDetector(model_file, device="cpu") as detector:
        detections = detector.detect(np.zeros((480, 640, 3), np.uint8), frame_index=7)
        feed = fake_ort.feeds[0]
        assert feed["output_names"] == ["output0"]
        assert feed["images"].shape == (1, 3, 640, 640) and feed["images"].dtype == np.float32
        assert len(detections) == 1
        d = detections[0]
        assert (d.frame_index, d.class_id, d.class_name) == (7, 2, "cat")
        assert d.confidence == pytest.approx(0.9)
        assert d.bbox_xyxy == pytest.approx((270, 210, 370, 270))  # 80 px of padding removed from y


def test_session_is_created_once(fake_ort, cv2, model_file: Path) -> None:
    detector = OnnxYoloDetector(model_file)
    frame = np.zeros((48, 64, 3), np.uint8)
    for i in range(5):
        assert detector.detect(frame, frame_index=i) == []  # empty output -> no detections
    assert len(fake_ort.sessions) == 1 and len(fake_ort.feeds) == 5


def test_detect_with_nms_output(fake_ort, cv2, model_file: Path) -> None:
    fake_ort.metadata["args"] = "{'nms': True}"
    fake_ort.outputs = [("output0", [1, 300, 6], "tensor(float)")]
    out = np.zeros((1, 300, 6), np.float32)
    out[0, 0] = [270, 290, 370, 350, 0.8, 1]
    out[0, 1] = [272, 292, 372, 352, 0.7, 1]  # overlapping: kept, NMS is not applied twice
    fake_ort.output = out
    detections = OnnxYoloDetector(model_file).detect(np.zeros((480, 640, 3), np.uint8))
    assert [d.class_name for d in detections] == ["bus", "bus"]
    assert detections[0].bbox_xyxy == pytest.approx((270, 210, 370, 270))


def test_inference_failure(fake_ort, cv2, model_file: Path) -> None:
    detector = OnnxYoloDetector(model_file)
    fake_ort.run_error = RuntimeError("Non-zero status code returned while running Conv node")
    with pytest.raises(InferenceError, match="failed on frame 3.*CPUExecutionProvider.*Conv node"):
        detector.detect(np.zeros((48, 64, 3), np.uint8), frame_index=3)


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (np.zeros((1, 9, 8400), np.float32), "4 box values \\+ 3 class scores"),
        (np.zeros((2, 7, 8400), np.float32), "batch size 1"),
        (np.zeros((7, 8400), np.float32), "3-D output"),
    ],
)
def test_unexpected_output_at_runtime(fake_ort, cv2, model_file: Path, output, message: str) -> None:
    fake_ort.output = output
    with pytest.raises(InferenceError, match=message):
        OnnxYoloDetector(model_file).detect(np.zeros((48, 64, 3), np.uint8))


def test_invalid_values_become_inference_errors(fake_ort, cv2, model_file: Path, make_raw_output) -> None:
    fake_ort.output = make_raw_output([(np.nan, 10, 5, 5, 0, 0.9)])
    with pytest.raises(InferenceError, match="invalid detection"):
        OnnxYoloDetector(model_file).detect(np.zeros((48, 64, 3), np.uint8))


def test_unknown_class_id_in_nms_output(fake_ort, cv2, model_file: Path) -> None:
    fake_ort.metadata["args"] = "{'nms': True}"
    fake_ort.outputs = [("output0", [1, 300, 6], "tensor(float)")]
    out = np.zeros((1, 300, 6), np.float32)
    out[0, 0] = [1, 1, 5, 5, 0.9, 42]
    fake_ort.output = out
    with pytest.raises(InferenceError, match="class ID 42"):
        OnnxYoloDetector(model_file).detect(np.zeros((48, 64, 3), np.uint8))


@pytest.mark.parametrize("frame", [np.zeros((48, 64), np.uint8), np.zeros((48, 64, 3), np.float32), "frame"])
def test_invalid_frames(fake_ort, cv2, model_file: Path, frame) -> None:
    with pytest.raises(ValueError):
        OnnxYoloDetector(model_file).detect(frame)


def test_closed_detector(fake_ort, cv2, model_file: Path) -> None:
    detector = OnnxYoloDetector(model_file)
    detector.close()
    detector.close()  # safe twice
    with pytest.raises(DetectorError, match="closed"):
        detector.detect(np.zeros((48, 64, 3), np.uint8))
    assert "closed" in repr(detector)


def test_nms_model_logs_that_iou_is_not_applied(fake_ort, cv2, model_file: Path, caplog) -> None:
    fake_ort.metadata["args"] = "{'nms': True}"
    fake_ort.outputs = [("output0", [1, 300, 6], "tensor(float)")]
    with caplog.at_level(logging.INFO):
        OnnxYoloDetector(model_file)
    assert "already contains NMS" in caplog.text


def test_importing_edgevision_does_not_import_onnxruntime() -> None:
    import subprocess

    code = (
        "import sys\n"
        "for name in ('onnxruntime', 'cv2', 'torch', 'ultralytics'):\n"
        "    sys.modules[name] = None\n"
        "import edgevision.models, edgevision.models.onnx_backend\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0 and result.stdout.strip() == "ok", result.stderr


# --- opt-in real model -------------------------------------------------------------------------


@pytest.mark.model
def test_real_onnx_model_matches_ultralytics_onnx_pipeline() -> None:
    """Real CPU smoke test: our pre/postprocessing vs Ultralytics running the same .onnx file.

    Both read the same ONNX model, so this checks EdgeVision's letterbox,
    decoding, NMS and box restoration, not PyTorch-vs-ONNX numerics.
    """
    if os.environ.get("EDGEVISION_RUN_MODEL_TESTS") != "1":
        pytest.skip("real-model test is opt-in: set EDGEVISION_RUN_MODEL_TESTS=1 (needs the vision, detector and "
                    "onnxruntime extras and models/yolov8n.onnx from scripts/export_onnx.py)")
    pytest.importorskip("onnxruntime")
    cv2_module = pytest.importorskip("cv2")
    ultralytics = pytest.importorskip("ultralytics")
    model = Path(os.environ.get("EDGEVISION_ONNX_MODEL", REPO / "models" / "yolov8n.onnx"))
    if not model.is_file():
        pytest.skip(f"{model} not found; create it with: python scripts/export_onnx.py")

    from ultralytics.utils import ASSETS

    image = cv2_module.imread(str(ASSETS / "bus.jpg"))
    with OnnxYoloDetector(model, device="cpu") as detector:
        assert detector.providers[0] == CPU and detector.device == "cpu"
        size = detector.layout.input_size or 640
        detector.settings = settings(input_size=size)
        ours = detector.detect(image)
    reference = ultralytics.YOLO(str(model), task="detect").predict(image, imgsz=size, device="cpu", verbose=False)[0]
    assert len(ours) == len(reference.boxes) > 0
    assert [d.class_id for d in ours] == [int(c) for c in reference.boxes.cls.tolist()]
    np.testing.assert_allclose([d.bbox_xyxy for d in ours], reference.boxes.xyxy.numpy(), atol=0.01)
    np.testing.assert_allclose([d.confidence for d in ours], reference.boxes.conf.numpy(), atol=1e-5)
    assert any(d.class_name == "bus" for d in ours) and any(d.class_name == "person" for d in ours)
