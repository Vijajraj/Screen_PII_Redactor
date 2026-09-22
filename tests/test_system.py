"""
test_system.py -- System-level tests exercising the full workflow end-to-end.

Tests the complete pipeline from:
  - Image file input -> Detection -> Recognition -> Classification -> Output
  - evaluate_pipeline.py execution and report generation
  - Model file integrity
  - Execution provider metadata
"""

import os
import subprocess
import sys

import onnx
import pytest

from inference_wrapper import PPOCRv4Detector, ScreenPIIPipeline

# =====================================================================
# MODEL FILE INTEGRITY
# =====================================================================


class TestModelIntegrity:
    """Verify model files exist and are structurally valid ONNX."""

    def test_quantized_detector_exists(self):
        assert os.path.exists("detector_quantized.onnx")
        assert os.path.getsize("detector_quantized.onnx") > 0

    def test_recognizer_model_exists(self):
        assert os.path.exists("models/rec/english/model.onnx")
        assert os.path.getsize("models/rec/english/model.onnx") > 0

    def test_dictionary_file_exists(self):
        assert os.path.exists("models/rec/english/dict.txt")
        with open("models/rec/english/dict.txt", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) >= 400, f"Dictionary too small: {len(lines)} lines"

    def test_quantized_detector_loads_as_valid_onnx(self):
        model = onnx.load("detector_quantized.onnx")
        onnx.checker.check_model(model)

    def test_detector_has_static_shape(self):
        model = onnx.load("detector_quantized.onnx")
        inp = model.graph.input[0]
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        assert shape == [1, 3, 640, 640], f"Unexpected shape: {shape}"

    def test_quantized_model_smaller_than_fp32(self):
        quant_size = os.path.getsize("detector_quantized.onnx")
        fp32_path = "models/detector_clean_static.onnx"
        if os.path.exists(fp32_path):
            fp32_size = os.path.getsize(fp32_path)
            assert quant_size < fp32_size, f"Quantized ({quant_size}) should be smaller than FP32 ({fp32_size})"


# =====================================================================
# FULL PIPELINE ROUND-TRIP
# =====================================================================


class TestFullPipelineRoundTrip:
    """End-to-end round-trip tests on the complete pipeline."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        return ScreenPIIPipeline()

    def test_pipeline_completes_without_error_on_all_images(self, pipeline):
        import glob

        images = glob.glob("synthetic_test_set/*.png")
        assert len(images) >= 20, f"Expected >= 20 images, found {len(images)}"
        for img_path in images:
            res = pipeline.run_on_image(img_path)
            assert isinstance(res, dict)
            assert "pii_findings" in res

    def test_pipeline_output_has_all_required_keys(self, pipeline):
        res = pipeline.run_on_image("synthetic_test_set/kyc_onboarding_01.png")
        required_keys = [
            "execution_provider",
            "total_text_regions",
            "detected_text_regions",
            "pii_findings",
        ]
        for key in required_keys:
            assert key in res, f"Missing key: {key}"

    def test_text_regions_have_complete_metadata(self, pipeline):
        res = pipeline.run_on_image("synthetic_test_set/banking_dashboard_01.png")
        for region in res["detected_text_regions"]:
            assert "bbox" in region
            assert "text" in region
            assert "det_confidence" in region
            assert "rec_confidence" in region
            assert len(region["bbox"]) == 4

    def test_pii_findings_have_complete_metadata(self, pipeline):
        res = pipeline.run_on_image("synthetic_test_set/kyc_onboarding_01.png")
        for finding in res["pii_findings"]:
            assert "pii_type" in finding
            assert "confidence" in finding
            assert "matched_text" in finding


# =====================================================================
# EVALUATION SCRIPT SYSTEM TEST
# =====================================================================


class TestEvaluationScript:
    """Verify evaluate_pipeline.py can be run as a script and produces output."""

    def test_evaluate_pipeline_runs_successfully(self):
        result = subprocess.run(
            [sys.executable, "evaluate_pipeline.py"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        assert result.returncode == 0, (
            f"evaluate_pipeline.py failed:\nSTDOUT: {result.stdout[:500]}\nSTDERR: {result.stderr[:500]}"
        )

    def test_results_file_generated(self):
        # Run evaluation first if needed
        if not os.path.exists("phase1_results.md"):
            subprocess.run(
                [sys.executable, "evaluate_pipeline.py"],
                capture_output=True,
                timeout=120,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )
        assert os.path.exists("phase1_results.md") or os.path.exists("results/phase1_results.md")


# =====================================================================
# EXECUTION PROVIDER SYSTEM TEST
# =====================================================================


class TestExecutionProviderSystem:
    """Verify EP abstraction works correctly at the system level."""

    def test_pipeline_reports_cpu_provider(self):
        pipeline = ScreenPIIPipeline()
        assert pipeline.execution_provider == "CPUExecutionProvider"

    def test_detector_metadata_is_consistent(self):
        detector = PPOCRv4Detector()
        meta = detector.provider_metadata
        assert meta["primary_provider"] == "CPUExecutionProvider"
        assert meta["is_cpu_fallback"] is True
        assert "CPUExecutionProvider" in meta["selected_providers"]
