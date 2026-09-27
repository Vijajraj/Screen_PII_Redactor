"""
test_phase3.py — Phase 3 Unit & Pipeline Verification Tests
Snapdragon AI Lab Challenge — Screen PII Redactor
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnx
import pytest

from evaluate_phase3 import (
    check_ai_hub_auth,
    generate_markdown_report,
    prepare_sample_input,
)


class TestPhase3ModelPreconditions:
    """Verifies that the ONNX model meets all Qualcomm AI Hub QNN compile requirements."""

    def test_detector_quantized_exists(self) -> None:
        assert os.path.exists("detector_quantized.onnx"), "detector_quantized.onnx must exist for Phase 3"

    def test_model_size_under_two_megabytes(self) -> None:
        size_mb = os.path.getsize("detector_quantized.onnx") / (1024 * 1024)
        assert size_mb < 2.0, f"Quantized model size ({size_mb:.2f} MB) should be < 2 MB"

    def test_input_shape_is_strictly_static_640(self) -> None:
        """QNN requires strict static dimensions to avoid CPU runtime fallbacks."""
        model = onnx.load("detector_quantized.onnx")
        input_tensor = model.graph.input[0]
        dims = [d.dim_value for d in input_tensor.type.tensor_type.shape.dim]
        assert dims == [1, 3, 640, 640], f"Expected static [1, 3, 640, 640], got {dims}"

    def test_output_shape_is_strictly_static_640(self) -> None:
        model = onnx.load("detector_quantized.onnx")
        output_tensor = model.graph.output[0]
        dims = [d.dim_value for d in output_tensor.type.tensor_type.shape.dim]
        assert dims == [1, 1, 640, 640], f"Expected static [1, 1, 640, 640], got {dims}"


class TestPhase3Preprocessing:
    """Verifies input preparation matches Phase 1 and Phase 2."""

    def test_prepare_sample_input_shape_and_dtype(self) -> None:
        sample_path = "synthetic_test_set/kyc_onboarding_01.png"
        tensor = prepare_sample_input(sample_path)
        assert isinstance(tensor, np.ndarray)
        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == np.float32

    def test_prepare_sample_input_normalization_range(self) -> None:
        sample_path = "synthetic_test_set/kyc_onboarding_01.png"
        tensor = prepare_sample_input(sample_path)
        # Normalized by ImageNet mean/std, values should generally fall between -3.0 and 3.0
        assert np.min(tensor) > -4.0
        assert np.max(tensor) < 4.0


class TestPhase3ReportGeneration:
    """Tests the markdown report generator produces complete proposal evidence."""

    def test_generate_markdown_report(self, tmp_path: Path) -> None:
        out_file = str(tmp_path / "test_npu_report.md")
        generate_markdown_report(
            output_path=out_file,
            device_str="Snapdragon X Elite CRD",
            compile_job_id="j_comp_12345",
            profile_job_id="j_prof_12345",
            inference_job_id="j_infer_12345",
            latency_ms=14.25,
            p95_latency_ms=18.50,
            npu_cycles_pct=100.0,
            cpu_cycles_pct=0.0,
            peak_memory_mb=18.6,
            mae_drift=0.0024,
            max_drift=0.015,
            correlation=0.9998,
            unquant_results={
                "fp32_latency_ms": 46.10,
                "fp32_peak_mem_mb": 42.1,
                "speedup_factor": 3.23,
                "size_reduction_pct": 72.0,
                "compile_job_id": "j_comp_fp32",
                "profile_job_id": "j_prof_fp32",
            },
        )
        assert os.path.exists(out_file)
        with open(out_file, encoding="utf-8") as f:
            content = f.read()

        assert "Snapdragon X Elite CRD" in content
        assert "14.25 ms" in content
        assert "100.0% NPU" in content
        assert "j_comp_12345" in content
        assert "3.23x speedup on Hexagon NPU" in content


class TestPhase3AuthCheck:
    """Tests authentication verification behavior."""

    def test_check_ai_hub_auth_returns_boolean(self) -> None:
        result = check_ai_hub_auth()
        assert isinstance(result, bool)
