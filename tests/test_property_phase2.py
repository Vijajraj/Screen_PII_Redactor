"""
test_property_phase2.py — Property-Based Testing for Phase 2 Components via Hypothesis
Snapdragon AI Lab Challenge — Screen PII Redactor

Validates invariants across:
1. Letterbox transformation across arbitrary dimensions and aspect ratios.
2. Coordinate projection invariants and boundary constraints.
3. RedactionRenderer resilience against arbitrary bounding boxes, labels, and kernel sizes.
4. HUD telemetry rendering across arbitrary parameter ranges.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from live_capture_app import LetterboxTransformer, RedactionRenderer

# Strategies
dimensions = st.integers(min_value=32, max_value=3840)
target_sizes = st.sampled_from([320, 480, 640, 800, 1024])
coordinates = st.integers(min_value=-500, max_value=4000)
pii_type_strategy = st.sampled_from(
    ["AADHAAR", "PAN", "CARD_NUMBER", "UPI_ID", "PHONE_IN", "IFSC", "EMAIL", "CUSTOM_PII", "UNKNOWN"]
)
confidences = st.floats(min_value=0.0, max_value=1.0)


# =====================================================================
# 1. LETTERBOX TRANSFORMER PROPERTY TESTS
# =====================================================================


class TestPropertyLetterbox:
    """Hypothesis invariant verification for LetterboxTransformer."""

    @given(w=dimensions, h=dimensions, target_size=target_sizes)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_letterbox_output_shape_invariant(self, w: int, h: int, target_size: int) -> None:
        """Invariant: transform() ALWAYS returns exact target_size x target_size canvas."""
        transformer = LetterboxTransformer(target_size=target_size)
        dummy = np.zeros((h, w, 3), dtype=np.uint8)

        canvas, meta = transformer.transform(dummy)

        assert canvas.shape == (target_size, target_size, 3)
        assert canvas.dtype == np.uint8
        assert meta["orig_w"] == w
        assert meta["orig_h"] == h
        assert meta["pad_x"] >= 0
        assert meta["pad_y"] >= 0
        assert meta["new_w"] <= target_size
        assert meta["new_h"] <= target_size

    @given(
        orig_w=st.integers(min_value=100, max_value=2500),
        orig_h=st.integers(min_value=100, max_value=2500),
        b_x1=coordinates,
        b_y1=coordinates,
        b_x2=coordinates,
        b_y2=coordinates,
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_coordinate_mapping_boundary_invariants(
        self, orig_w: int, orig_h: int, b_x1: int, b_y1: int, b_x2: int, b_y2: int
    ) -> None:
        """Invariant: mapped bounding box coordinates are ALWAYS clamped to [0, W] and [0, H]."""
        transformer = LetterboxTransformer(target_size=640)
        dummy = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        _canvas, meta = transformer.transform(dummy)

        box_input = [b_x1, b_y1, b_x2, b_y2]
        mapped = transformer.map_box_to_original(box_input, meta)

        assert len(mapped) == 4
        x1, y1, x2, y2 = mapped

        assert 0 <= x1 <= orig_w
        assert 0 <= y1 <= orig_h
        assert 0 <= x2 <= orig_w
        assert 0 <= y2 <= orig_h
        assert x1 <= x2
        assert y1 <= y2


# =====================================================================
# 2. REDACTION RENDERER RESILIENCE PROPERTY TESTS
# =====================================================================


class TestPropertyRedactionRenderer:
    """Hypothesis invariant verification for RedactionRenderer."""

    @given(
        x1=coordinates,
        y1=coordinates,
        x2=coordinates,
        y2=coordinates,
        p_type=pii_type_strategy,
        conf=confidences,
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_apply_redactions_never_crashes_on_arbitrary_boxes(
        self, x1: int, y1: int, x2: int, y2: int, p_type: str, conf: float
    ) -> None:
        """Invariant: apply_redactions never crashes and preserves frame dimensions and dtype."""
        frame = np.ones((300, 400, 3), dtype=np.uint8) * 128
        finding = {
            "bbox": [x1, y1, x2, y2],
            "pii_type": p_type,
            "confidence": conf,
            "matched_text": "SAMPLE_PII",
        }

        output = RedactionRenderer.apply_redactions(frame, [finding])

        assert isinstance(output, np.ndarray)
        assert output.shape == frame.shape
        assert output.dtype == np.uint8

    @given(
        latency=st.floats(min_value=-100.0, max_value=10000.0),
        fps=st.floats(min_value=-50.0, max_value=240.0),
        count=st.integers(min_value=-10, max_value=500),
        interval=st.floats(min_value=0.01, max_value=10.0),
        provider=st.text(min_size=0, max_size=30),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_render_hud_never_crashes(
        self, latency: float, fps: float, count: int, interval: float, provider: str
    ) -> None:
        """Invariant: render_hud accepts arbitrary telemetry inputs without crashing."""
        frame = np.zeros((300, 500, 3), dtype=np.uint8)
        output = RedactionRenderer.render_hud(
            frame,
            provider=provider,
            latency_ms=latency,
            fps=fps,
            detected_count=count,
            interval_s=interval,
        )

        assert isinstance(output, np.ndarray)
        assert output.shape == frame.shape
        assert output.dtype == np.uint8
