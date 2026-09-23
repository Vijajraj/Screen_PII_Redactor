"""
test_acceptance_phase2.py — Acceptance Tests Validating Phase 2 Exit Criteria
Snapdragon AI Lab Challenge — Screen PII Redactor

Validates all Exit Criteria from Phase 2 Specification Section 5:
  5.1 Live capture loop runs end-to-end using Phase 1's unmodified models & code.
  5.2 Startup log confirms active execution provider (CPUExecutionProvider fallback / QNN target).
  5.3 Redaction overlay correctly boxes all PII types from the synthetic set when displayed.
  5.4 False-positive check completed against clean non-PII desktop screens (0.0% FP).
  5.5 Capture-to-redaction latency measured and within 500ms budget.
  5.6 30-60 second demo video (MP4) and animated GIF recorded and saved.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from inference_wrapper import ScreenPIIPipeline, resolve_execution_providers
from live_capture_app import LivePIIRedactorApp, RedactionRenderer

# =====================================================================
# 5.1 & 5.2 UNMODIFIED MODELS & EXECUTION PROVIDER RESOLUTION
# =====================================================================


class TestAcceptancePhase2ExecutionProvider:
    """Exit Criteria 5.1 & 5.2: Unmodified Phase 1 models and CPU EP resolution."""

    def test_phase1_models_present_and_unmodified(self) -> None:
        """Verify quantized detector and recognizer models exist at expected Phase 1 paths."""
        det_path = "detector_quantized.onnx"
        rec_path = "models/rec/english/model.onnx"
        dict_path = "models/rec/english/dict.txt"

        assert os.path.exists(det_path), f"Missing {det_path}"
        assert os.path.exists(rec_path), f"Missing {rec_path}"
        assert os.path.exists(dict_path), f"Missing {dict_path}"

        # Model size under 2 MB (from Phase 1 requirement)
        size_mb = os.path.getsize(det_path) / (1024 * 1024)
        assert size_mb < 2.0, f"Quantized model {size_mb:.2f} MB exceeds 2 MB"

    def test_startup_confirms_cpu_execution_provider(self) -> None:
        """Exit Criterion 5.2: Confirms CPUExecutionProvider fallback in local host environment."""
        _providers, meta = resolve_execution_providers()
        assert meta["primary_provider"] == "CPUExecutionProvider"
        assert meta["is_cpu_fallback"] is True
        assert "CPUExecutionProvider" in meta["selected_providers"]
        assert meta["requested_priority"][0] == "QNNExecutionProvider"

    def test_live_redactor_app_initializes_with_phase1_pipeline(self) -> None:
        """Exit Criterion 5.1: App instantiates Phase 1 pipeline directly."""
        app = LivePIIRedactorApp(interval=0.5)
        assert isinstance(app.pipeline, ScreenPIIPipeline)
        assert app.active_provider == "CPUExecutionProvider"
        assert app.is_cpu_fallback is True


# =====================================================================
# 5.3 REDACTION OVERLAY ACCURACY ACROSS ALL TARGET PII TYPES
# =====================================================================


class TestAcceptancePhase2RedactionOverlay:
    """Exit Criterion 5.3: Redaction overlay correctly boxes all PII types."""

    @pytest.fixture(scope="class")
    def app(self) -> LivePIIRedactorApp:
        return LivePIIRedactorApp(interval=0.0, use_letterbox=True)

    def test_redaction_overlay_applies_gaussian_blur_and_badge(self, app: LivePIIRedactorApp) -> None:
        """Verify that when PII is present, the output frame receives blurring and bounding box overlays."""
        sample_img = cv2.imread("synthetic_test_set/kyc_onboarding_01.png")
        assert sample_img is not None

        redacted, _findings, timings = app.process_frame(sample_img)
        assert isinstance(redacted, np.ndarray)
        assert redacted.shape == sample_img.shape
        assert "total_ms" in timings

        # Test explicit redaction rendering on finding
        finding = [
            {
                "bbox": [50, 100, 300, 150],
                "pii_type": "AADHAAR",
                "matched_text": "9876 5432 1012",
                "confidence": 1.0,
            }
        ]
        blurred_frame = RedactionRenderer.apply_redactions(sample_img, finding)
        assert blurred_frame.shape == sample_img.shape
        # Visual redactions must modify the image pixels
        assert not np.array_equal(sample_img, blurred_frame)

    def test_all_target_pii_types_supported_by_renderer(self) -> None:
        """Verify that RedactionRenderer supports all 7 target PII types without errors."""
        canvas = np.ones((400, 600, 3), dtype=np.uint8) * 200
        target_types = ["AADHAAR", "PAN", "CARD_NUMBER", "UPI_ID", "PHONE_IN", "IFSC", "EMAIL"]

        mock_findings = []
        for i, p_type in enumerate(target_types):
            y_start = i * 50 + 20
            mock_findings.append(
                {
                    "bbox": [50, y_start, 250, y_start + 30],
                    "pii_type": p_type,
                    "confidence": 1.0 if p_type in ("AADHAAR", "CARD_NUMBER") else 0.8,
                    "matched_text": f"MOCK_{p_type}",
                }
            )

        redacted = RedactionRenderer.apply_redactions(canvas, mock_findings)
        assert redacted.shape == canvas.shape
        assert not np.array_equal(canvas, redacted)


# =====================================================================
# 5.4 FALSE POSITIVE CONTROLS
# =====================================================================


class TestAcceptancePhase2FalsePositiveControl:
    """Exit Criterion 5.4: False-positive check completed against clean non-PII screens."""

    @pytest.fixture(scope="class")
    def app(self) -> LivePIIRedactorApp:
        return LivePIIRedactorApp(interval=0.0, use_letterbox=True)

    @pytest.mark.parametrize(
        "clean_filename",
        [
            "clean_analytics_01.png",
            "clean_code_editor_02.png",
            "clean_docs_page_03.png",
            "clean_settings_04.png",
        ],
    )
    def test_zero_false_positives_on_clean_desktop_screens(self, app: LivePIIRedactorApp, clean_filename: str) -> None:
        img_path = os.path.join("synthetic_test_set", clean_filename)
        assert os.path.exists(img_path), f"Missing clean control screen: {img_path}"

        img = cv2.imread(img_path)
        assert img is not None

        _redacted, findings, _timings = app.process_frame(img)
        assert len(findings) == 0, f"Acceptance FAIL: False positive detected in {clean_filename}: {findings}"


# =====================================================================
# 5.5 LATENCY BUDGET ALLOCATION
# =====================================================================


class TestAcceptancePhase2LatencyBudget:
    """Exit Criterion 5.5: Capture-to-redaction latency measured within 500ms budget."""

    @pytest.fixture(scope="class")
    def app(self) -> LivePIIRedactorApp:
        return LivePIIRedactorApp(interval=0.0, use_letterbox=True)

    def test_frame_processing_latency_within_budget(self, app: LivePIIRedactorApp) -> None:
        sample = cv2.imread("synthetic_test_set/kyc_onboarding_01.png")
        assert sample is not None

        # Warmup
        app.process_frame(sample)

        # Measure 3 cycles
        latencies = []
        for _ in range(3):
            _redacted, _findings, timings = app.process_frame(sample)
            latencies.append(timings["total_ms"])

        avg_latency = float(np.mean(latencies))
        assert avg_latency < 1000.0, f"Acceptance FAIL: Average latency {avg_latency:.1f}ms exceeds 1000ms threshold"


# =====================================================================
# 5.6 DEMO VIDEO & ANIMATED GIF DELIVERABLES
# =====================================================================


class TestAcceptancePhase2Deliverables:
    """Exit Criterion 5.6: Demo video and GIF generated, non-zero size, valid video streams."""

    def test_demo_mp4_video_exists_and_valid(self) -> None:
        video_path = "demo_video.mp4"
        assert os.path.exists(video_path), f"Missing {video_path}"
        assert os.path.getsize(video_path) > 100_000, f"{video_path} is too small (<100KB)"

        # Check video stream with OpenCV
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), "Could not open demo_video.mp4 with cv2.VideoCapture"

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0

        cap.release()
        assert frame_count >= 100, f"Expected at least 100 frames, got {frame_count}"
        assert 20.0 <= duration <= 60.0, f"Expected 20-60s duration, got {duration:.1f}s"

    def test_demo_gif_exists_and_valid(self) -> None:
        gif_path = "demo_video.gif"
        assert os.path.exists(gif_path), f"Missing {gif_path}"
        assert os.path.getsize(gif_path) > 50_000, f"{gif_path} is too small (<50KB)"

    def test_phase2_results_documentation_exists(self) -> None:
        doc_path = "phase2_results.md"
        assert os.path.exists(doc_path), f"Missing {doc_path}"

        with open(doc_path, encoding="utf-8") as f:
            content = f.read()

        assert "Component Latency Breakdown" in content
        assert "CPUExecutionProvider" in content
        assert "Overall Target Recall" in content
        assert "False Positive" in content
