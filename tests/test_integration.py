"""
test_integration.py -- Integration tests verifying component interactions.

Tests the seams between:
  - Detector -> Recognizer (crop hand-off)
  - Recognizer -> Classifier (text hand-off)
  - Detector -> Recognizer -> Classifier (two-component chain)
  - PIL Image input -> Pipeline
  - numpy array input -> Pipeline
  - Multiple image categories through the pipeline
"""

import glob
import json
import os

import cv2
import pytest
from PIL import Image

from inference_wrapper import (
    PPOCRv4Detector,
    PPOCRv4Recognizer,
    ScreenPIIPipeline,
)
from pii_classifier import classify_text

# =====================================================================
# DETECTOR <-> RECOGNIZER INTEGRATION
# =====================================================================


class TestDetectorRecognizerIntegration:
    """Verify detector output can be consumed by the recognizer."""

    @pytest.fixture(scope="class")
    def detector(self):
        return PPOCRv4Detector(model_path="detector_quantized.onnx")

    @pytest.fixture(scope="class")
    def recognizer(self):
        return PPOCRv4Recognizer()

    def test_detector_boxes_are_valid_crops_for_recognizer(self, detector, recognizer):
        img = cv2.imread("synthetic_test_set/kyc_onboarding_01.png")
        boxes = detector.detect(img)
        assert len(boxes) > 0, "Detector must find text regions"

        for b in boxes:
            x1, y1, x2, y2 = b["bbox"]
            crop = img[y1:y2, x1:x2]
            assert crop.shape[0] > 0 and crop.shape[1] > 0, f"Empty crop from bbox {b['bbox']}"
            text, conf = recognizer.recognize_crop(crop)
            assert isinstance(text, str)
            assert isinstance(conf, float)

    def test_recognized_text_is_classifiable(self, detector, recognizer):
        """Whatever text the recognizer emits can be fed to classify_text
        without errors, even if PII is not found."""
        img = cv2.imread("synthetic_test_set/email_client_01.png")
        boxes = detector.detect(img)
        for b in boxes:
            x1, y1, x2, y2 = b["bbox"]
            crop = img[y1:y2, x1:x2]
            text, _ = recognizer.recognize_crop(crop)
            # Must not raise
            results = classify_text(text)
            assert isinstance(results, list)


# =====================================================================
# PIPELINE INPUT FORMAT INTEGRATION
# =====================================================================


class TestPipelineInputFormats:
    """Verify the pipeline accepts all documented input types."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        return ScreenPIIPipeline()

    def test_accepts_file_path_string(self, pipeline):
        res = pipeline.run_on_image("synthetic_test_set/kyc_onboarding_01.png")
        assert "detected_text_regions" in res
        assert "pii_findings" in res

    def test_accepts_numpy_array(self, pipeline):
        img = cv2.imread("synthetic_test_set/kyc_onboarding_01.png")
        res = pipeline.run_on_image(img)
        assert "detected_text_regions" in res

    def test_accepts_pil_image(self, pipeline):
        pil_img = Image.open("synthetic_test_set/kyc_onboarding_01.png")
        res = pipeline.run_on_image(pil_img)
        assert "detected_text_regions" in res

    def test_rejects_invalid_path(self, pipeline):
        with pytest.raises(ValueError, match="Could not load image"):
            pipeline.run_on_image("nonexistent_file.png")

    def test_rejects_unsupported_type(self, pipeline):
        with pytest.raises(TypeError, match="Unsupported image input type"):
            pipeline.run_on_image(12345)


# =====================================================================
# CROSS-CATEGORY INTEGRATION
# =====================================================================


class TestCrossCategoryIntegration:
    """Run the pipeline on each synthetic category and verify structural output."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        return ScreenPIIPipeline()

    @pytest.mark.parametrize("image_path", sorted(glob.glob("synthetic_test_set/*.png")))
    def test_pipeline_returns_valid_structure(self, pipeline, image_path):
        res = pipeline.run_on_image(image_path)
        assert "execution_provider" in res
        assert "total_text_regions" in res
        assert "detected_text_regions" in res
        assert "pii_findings" in res
        assert isinstance(res["detected_text_regions"], list)
        assert isinstance(res["pii_findings"], list)
        assert res["total_text_regions"] == len(res["detected_text_regions"])

    def test_clean_images_produce_zero_pii(self, pipeline):
        clean_files = glob.glob("synthetic_test_set/clean_*.png")
        assert len(clean_files) >= 4, "Expected at least 4 clean images"
        for path in clean_files:
            res = pipeline.run_on_image(path)
            assert len(res["pii_findings"]) == 0, f"False positive on clean image: {path}"


# =====================================================================
# GROUND TRUTH ALIGNMENT
# =====================================================================


class TestGroundTruthAlignment:
    """Verify ground truth JSON aligns with generated images."""

    def test_ground_truth_file_exists(self):
        assert os.path.exists("synthetic_test_set/ground_truth.json")

    def test_all_referenced_images_exist(self):
        with open("synthetic_test_set/ground_truth.json") as f:
            gt = json.load(f)
        for entry in gt:
            path = os.path.join("synthetic_test_set", entry["file"])
            assert os.path.exists(path), f"Missing image: {path}"

    def test_clean_entries_have_zero_ground_truth(self):
        with open("synthetic_test_set/ground_truth.json") as f:
            gt = json.load(f)
        for entry in gt:
            if entry["is_clean"]:
                assert entry["ground_truth_count"] == 0
                assert len(entry["ground_truth_items"]) == 0

    def test_pii_entries_have_ground_truth(self):
        with open("synthetic_test_set/ground_truth.json") as f:
            gt = json.load(f)
        for entry in gt:
            if not entry["is_clean"]:
                assert entry["ground_truth_count"] > 0
                assert len(entry["ground_truth_items"]) > 0
