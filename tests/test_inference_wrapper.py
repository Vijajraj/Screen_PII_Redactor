"""
test_inference_wrapper.py — Unit tests for EP abstraction and ONNX detector
Phase 1 — Screen PII Redactor (Snapdragon AI Lab Challenge)
"""

import os

import onnx

from inference_wrapper import TARGET_PROVIDER_PRIORITY, PPOCRv4Detector, ScreenPIIPipeline, resolve_execution_providers


def test_execution_provider_resolution_and_fallback():
    _providers, meta = resolve_execution_providers()
    assert meta["requested_priority"] == TARGET_PROVIDER_PRIORITY
    assert "CPUExecutionProvider" in meta["selected_providers"]
    assert meta["primary_provider"] == "CPUExecutionProvider"
    assert meta["is_cpu_fallback"] is True


def test_quantized_model_file_and_static_shape():
    model_path = "detector_quantized.onnx"
    assert os.path.exists(model_path)

    m = onnx.load(model_path)
    inputs = m.graph.input
    assert len(inputs) == 1
    shape_dims = [d.dim_value for d in inputs[0].type.tensor_type.shape.dim]
    assert shape_dims == [1, 3, 640, 640], f"Expected static [1, 3, 640, 640], got {shape_dims}"


def test_detector_inference_on_synthetic_screenshot():
    detector = PPOCRv4Detector(model_path="detector_quantized.onnx")
    assert detector.provider_metadata["primary_provider"] == "CPUExecutionProvider"

    # Run on sample synthetic screenshot
    test_img_path = "synthetic_test_set/kyc_onboarding_01.png"
    assert os.path.exists(test_img_path)

    import cv2

    img = cv2.imread(test_img_path)
    boxes = detector.detect(img)
    assert len(boxes) > 0
    for b in boxes:
        assert "bbox" in b
        assert len(b["bbox"]) == 4
        x1, y1, x2, y2 = b["bbox"]
        assert x1 < x2
        assert y1 < y2
        assert 0.0 <= b["det_confidence"] <= 1.0


def test_pipeline_on_clean_image():
    pipeline = ScreenPIIPipeline()
    clean_img_path = "synthetic_test_set/clean_analytics_01.png"
    assert os.path.exists(clean_img_path)

    res = pipeline.run_on_image(clean_img_path)
    assert res["execution_provider"] == "CPUExecutionProvider"
    assert "detected_text_regions" in res
    assert "pii_findings" in res
    # Clean image should have 0 PII findings
    assert len(res["pii_findings"]) == 0
