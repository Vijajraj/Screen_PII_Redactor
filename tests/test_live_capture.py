"""
test_live_capture.py — Comprehensive Unit & Integration Tests for Phase 2
Snapdragon AI Lab Challenge — Screen PII Redactor

Tests:
1. Letterbox transformation and bidirectional coordinate mapping.
2. ScreenCaptureSource frame acquisition and headless fallback.
3. RedactionRenderer Gaussian blur, visual badges, and HUD telemetry.
4. LivePIIRedactorApp end-to-end processing and timing profiling.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from live_capture_app import (
    LetterboxTransformer,
    LivePIIRedactorApp,
    RedactionRenderer,
    ScreenCaptureSource,
)

# =====================================================================
# 1. LETTERBOX TRANSFORMER TESTS
# =====================================================================


class TestLetterboxTransformer:
    """Verifies aspect-ratio preserving letterboxing and coordinate mapping."""

    @pytest.fixture
    def transformer(self) -> LetterboxTransformer:
        return LetterboxTransformer(target_size=640)

    @pytest.mark.parametrize(
        "orig_w,orig_h",
        [
            (1920, 1080),  # 16:9 widescreen
            (2560, 1440),  # 2K 16:9
            (1280, 1024),  # 5:4 legacy
            (800, 600),  # 4:3
            (640, 640),  # 1:1 square
            (1080, 1920),  # 9:16 vertical / phone
        ],
    )
    def test_transform_output_shape_always_static_640(
        self, transformer: LetterboxTransformer, orig_w: int, orig_h: int
    ) -> None:
        dummy = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        boxed, meta = transformer.transform(dummy)

        assert boxed.shape == (640, 640, 3)
        assert meta["orig_w"] == orig_w
        assert meta["orig_h"] == orig_h
        assert meta["new_w"] <= 640
        assert meta["new_h"] <= 640
        assert meta["pad_x"] >= 0
        assert meta["pad_y"] >= 0

    def test_coordinate_mapping_round_trip(self, transformer: LetterboxTransformer) -> None:
        """Verify that mapping a box from letterbox coordinates lands accurately in original space."""
        orig_w, orig_h = 1920, 1080
        dummy = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        _boxed, meta = transformer.transform(dummy)

        # Scale is 640 / 1920 = 1/3, pad_x = 0, pad_y = (640 - 360) / 2 = 140
        # A box at [0, 140, 640, 500] in 640x640 space should map to [0, 0, 1920, 1080]
        box_640 = [0, 140, 640, 500]
        mapped = transformer.map_box_to_original(box_640, meta)

        assert mapped[0] == 0
        assert mapped[1] == 0
        assert mapped[2] == orig_w
        assert mapped[3] == orig_h

    def test_coordinate_clamping(self, transformer: LetterboxTransformer) -> None:
        """Verify coordinates outside the image pad region clamp to valid bounds."""
        orig_w, orig_h = 1000, 500
        dummy = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        _boxed, meta = transformer.transform(dummy)

        out_of_bounds_box = [-50, -50, 700, 700]
        mapped = transformer.map_box_to_original(out_of_bounds_box, meta)

        assert mapped[0] >= 0
        assert mapped[1] >= 0
        assert mapped[2] <= orig_w
        assert mapped[3] <= orig_h


# =====================================================================
# 2. SCREEN CAPTURE SOURCE TESTS
# =====================================================================


class TestScreenCaptureSource:
    """Verifies screen capture initialization, frame format, and headless fallback."""

    def test_capture_initialization_and_dimensions(self) -> None:
        source = ScreenCaptureSource()
        w, h = source.get_dimensions()
        assert w > 0
        assert h > 0
        source.close()

    def test_grab_frame_returns_valid_bgr_array(self) -> None:
        source = ScreenCaptureSource()
        frame = source.grab_frame()

        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        assert frame.dtype == np.uint8
        source.close()

    def test_region_capture(self) -> None:
        region = {"left": 0, "top": 0, "width": 400, "height": 300}
        source = ScreenCaptureSource(region=region)
        w, h = source.get_dimensions()
        assert (w, h) == (400, 300)

        frame = source.grab_frame()
        assert frame.shape[1] == 400
        assert frame.shape[0] == 300
        source.close()


# =====================================================================
# 3. REDACTION RENDERER TESTS
# =====================================================================


class TestRedactionRenderer:
    """Verifies Gaussian blur redaction, badges, and HUD telemetry banner."""

    def test_apply_redactions_blurs_target_region(self) -> None:
        # Create a test frame with high contrast text
        frame = np.ones((200, 400, 3), dtype=np.uint8) * 255
        cv2.putText(frame, "9876 5432 1012", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Record variance before redaction
        target_roi = frame[70:120, 40:300]
        var_before = float(np.var(target_roi))
        assert var_before > 100.0, "ROI should have text variation before blur"

        finding = {
            "bbox": [40, 70, 300, 120],
            "pii_type": "AADHAAR",
            "matched_text": "9876 5432 1012",
            "confidence": 1.0,
        }
        redacted = RedactionRenderer.apply_redactions(frame, [finding], blur_kernel_size=25)

        # Redacted frame must have the same shape
        assert redacted.shape == frame.shape
        # The blurred region should differ from original
        assert not np.array_equal(redacted[70:120, 40:300], target_roi)

    def test_empty_findings_returns_identical_frame(self) -> None:
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        redacted = RedactionRenderer.apply_redactions(frame, [])
        assert np.array_equal(frame, redacted)

    def test_render_hud_preserves_dimensions(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hud_frame = RedactionRenderer.render_hud(
            frame,
            provider="CPUExecutionProvider",
            latency_ms=185.4,
            fps=5.4,
            detected_count=2,
            interval_s=0.5,
        )
        assert hud_frame.shape == (480, 640, 3)


# =====================================================================
# 4. LIVE PII REDACTOR APP END-TO-END TESTS
# =====================================================================


class TestLivePIIRedactorApp:
    """Verifies end-to-end processing pipeline, provider resolution, and timings."""

    @pytest.fixture(scope="class")
    def app(self) -> LivePIIRedactorApp:
        return LivePIIRedactorApp(interval=0.5, use_letterbox=True)

    def test_provider_is_cpu_execution_provider(self, app: LivePIIRedactorApp) -> None:
        assert app.active_provider == "CPUExecutionProvider"
        assert app.is_cpu_fallback is True

    def test_process_frame_returns_complete_structure(self, app: LivePIIRedactorApp) -> None:
        test_img = cv2.imread("synthetic_test_set/kyc_onboarding_01.png")
        assert test_img is not None

        redacted, findings, timings = app.process_frame(test_img)

        assert isinstance(redacted, np.ndarray)
        assert redacted.shape == test_img.shape
        assert isinstance(findings, list)

        assert "preprocess_ms" in timings
        assert "inference_ms" in timings
        assert "render_ms" in timings
        assert "total_ms" in timings
        assert timings["total_ms"] > 0

    def test_clean_image_produces_zero_findings(self, app: LivePIIRedactorApp) -> None:
        clean_img = cv2.imread("synthetic_test_set/clean_analytics_01.png")
        assert clean_img is not None

        _redacted, findings, _timings = app.process_frame(clean_img)
        assert len(findings) == 0, f"Expected 0 findings on clean screen, got {findings}"
