"""
test_acceptance.py -- Acceptance tests validating Phase 1 exit criteria.

From Section 7 of the Phase 1 Spec:
  7.1  OCR produces correct bounding boxes on synthetic screenshots
  7.2  PII classifier achieves >90% recall on India-format PII types
  7.3  Zero false positives on clean control images
  7.4  ONNX export has fixed static input shape [1, 3, 640, 640]
  7.5  Quantized INT8 model size < 2 MB
  7.6  Quantized model output drift (MAE) < 0.01 compared to FP32
  7.7  EP wrapper correctly selects CPU fallback
  7.8  Pipeline output JSON has required schema
"""

import glob
import json
import os

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from inference_wrapper import (
    PPOCRv4Detector,
    ScreenPIIPipeline,
    resolve_execution_providers,
)
from pii_classifier import classify_text

# =====================================================================
# 7.1 OCR BOUNDING BOXES
# =====================================================================


class TestAcceptanceBoundingBoxes:
    """OCR produces text bounding boxes on synthetic screenshots."""

    @pytest.fixture(scope="class")
    def detector(self):
        return PPOCRv4Detector(model_path="detector_quantized.onnx")

    def test_detector_finds_text_in_pii_images(self, detector):
        pii_images = [f for f in glob.glob("synthetic_test_set/*.png") if "clean" not in os.path.basename(f)]
        assert len(pii_images) >= 16
        for img_path in pii_images:
            img = cv2.imread(img_path)
            boxes = detector.detect(img)
            assert len(boxes) > 0, f"Acceptance FAIL: No text detected in {img_path}"

    def test_bounding_boxes_are_within_image_bounds(self, detector):
        img_path = "synthetic_test_set/kyc_onboarding_01.png"
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        boxes = detector.detect(img)
        for b in boxes:
            x1, y1, x2, y2 = b["bbox"]
            assert 0 <= x1 < w, f"x1 out of bounds: {x1}"
            assert 0 <= y1 < h, f"y1 out of bounds: {y1}"
            assert x1 < x2 <= w, f"x2 out of bounds or <= x1: {x2}"
            assert y1 < y2 <= h, f"y2 out of bounds or <= y1: {y2}"


# =====================================================================
# 7.2 PII RECALL >90%
# =====================================================================


class TestAcceptancePIIRecall:
    """PII classifier achieves >90% recall on India-format PII."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        return ScreenPIIPipeline()

    def test_recall_exceeds_ninety_percent(self):
        with open("synthetic_test_set/ground_truth.json", encoding="utf-8") as f:
            gt = json.load(f)

        total_gt = 0
        total_detected = 0

        for entry in gt:
            if entry["is_clean"]:
                continue
            for item in entry["ground_truth_items"]:
                total_gt += 1
                gt_type = item["pii_type"]
                text_val = item["text"]
                hits = classify_text(text_val)
                matched_types = [h["pii_type"] for h in hits]
                if gt_type in matched_types:
                    total_detected += 1

        recall = total_detected / total_gt if total_gt > 0 else 0
        assert recall >= 0.90, f"Acceptance FAIL: PII recall {recall:.1%} < 90% ({total_detected}/{total_gt})"


# =====================================================================
# 7.3 ZERO FALSE POSITIVES ON CLEAN IMAGES
# =====================================================================


class TestAcceptanceZeroFP:
    """Zero false positives on clean control images."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        return ScreenPIIPipeline()

    def test_clean_images_have_zero_pii(self, pipeline):
        clean_files = glob.glob("synthetic_test_set/clean_*.png")
        assert len(clean_files) >= 4, "Need at least 4 clean control images"
        for path in clean_files:
            res = pipeline.run_on_image(path)
            assert len(res["pii_findings"]) == 0, (
                f"Acceptance FAIL: False positive on {os.path.basename(path)}: {res['pii_findings']}"
            )


# =====================================================================
# 7.4 ONNX STATIC SHAPE
# =====================================================================


class TestAcceptanceONNXShape:
    """ONNX export has fixed static input shape [1, 3, 640, 640]."""

    def test_static_input_shape(self):
        model = onnx.load("detector_quantized.onnx")
        inp = model.graph.input[0]
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        assert shape == [1, 3, 640, 640], f"Acceptance FAIL: Expected static [1,3,640,640], got {shape}"


# =====================================================================
# 7.5 QUANTIZED MODEL SIZE < 2 MB
# =====================================================================


class TestAcceptanceModelSize:
    """Quantized INT8 model is under 2 MB."""

    def test_quantized_model_under_2mb(self):
        size_bytes = os.path.getsize("detector_quantized.onnx")
        size_mb = size_bytes / (1024 * 1024)
        assert size_mb < 2.0, f"Acceptance FAIL: Quantized model {size_mb:.2f} MB >= 2 MB"


# =====================================================================
# 7.6 QUANTIZATION DRIFT < 0.01
# =====================================================================


class TestAcceptanceQuantizationDrift:
    """Quantized model output drift (MAE) < 0.01 compared to FP32."""

    def test_output_drift_within_tolerance(self):
        fp32_path = "models/detector_clean_static.onnx"
        int8_path = "detector_quantized.onnx"

        if not os.path.exists(fp32_path):
            pytest.skip("FP32 reference model not available for drift test")

        fp32_session = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
        int8_session = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])

        # Synthetic input
        dummy = np.random.randn(1, 3, 640, 640).astype(np.float32)
        input_name = fp32_session.get_inputs()[0].name

        fp32_out = fp32_session.run(None, {input_name: dummy})[0]
        int8_out = int8_session.run(None, {input_name: dummy})[0]

        mae = np.mean(np.abs(fp32_out - int8_out))
        assert mae < 0.01, f"Acceptance FAIL: Quantization drift MAE={mae:.6f} >= 0.01"


# =====================================================================
# 7.7 EP WRAPPER CPU FALLBACK
# =====================================================================


class TestAcceptanceEPWrapper:
    """EP wrapper correctly selects CPU fallback on non-NPU hardware."""

    def test_cpu_fallback_selected(self):
        _providers, meta = resolve_execution_providers()
        assert meta["primary_provider"] == "CPUExecutionProvider"
        assert meta["is_cpu_fallback"] is True


# =====================================================================
# 7.8 PIPELINE OUTPUT SCHEMA
# =====================================================================


class TestAcceptanceOutputSchema:
    """Pipeline output JSON has the required schema."""

    def test_output_has_all_required_fields(self):
        pipeline = ScreenPIIPipeline()
        res = pipeline.run_on_image("synthetic_test_set/kyc_onboarding_01.png")

        # Top-level keys
        assert "execution_provider" in res
        assert "total_text_regions" in res
        assert "detected_text_regions" in res
        assert "pii_findings" in res

        # Type checks
        assert isinstance(res["execution_provider"], str)
        assert isinstance(res["total_text_regions"], int)
        assert isinstance(res["detected_text_regions"], list)
        assert isinstance(res["pii_findings"], list)

        # Region schema
        if res["detected_text_regions"]:
            region = res["detected_text_regions"][0]
            assert "bbox" in region
            assert "text" in region
            assert "det_confidence" in region
            assert "rec_confidence" in region

        # PII finding schema
        if res["pii_findings"]:
            finding = res["pii_findings"][0]
            assert "pii_type" in finding
            assert "confidence" in finding
            assert "matched_text" in finding
