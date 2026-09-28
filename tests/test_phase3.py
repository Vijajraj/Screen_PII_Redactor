"""
test_phase3.py — Phase 3 Unit & Pipeline Verification Tests
Snapdragon AI Lab Challenge — Screen PII Redactor
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import onnx

from evaluate_phase3 import (
    check_ai_hub_auth,
    compile_model_qnn,
    extract_profile_metrics,
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

    def test_generate_markdown_report_with_compile_meta(self, tmp_path: Path) -> None:
        out_file = str(tmp_path / "test_modern_report.md")
        compile_meta = {
            "compile_job_id": "j_dlc_001",
            "link_job_id": "j_link_002",
            "embed_job_id": "j_embed_003",
            "primary_job_id": "j_embed_003",
            "api_used": "submit_compile_and_link_jobs",
        }
        generate_markdown_report(
            output_path=out_file,
            device_str="Snapdragon X Elite CRD",
            compile_job_id="j_embed_003",
            profile_job_id="j_prof_12345",
            inference_job_id="j_infer_12345",
            latency_ms=15.96,
            p95_latency_ms=16.27,
            npu_cycles_pct=100.0,
            cpu_cycles_pct=0.0,
            peak_memory_mb=36.90,
            mae_drift=0.0023,
            max_drift=0.012,
            correlation=0.9745,
            compile_meta=compile_meta,
        )
        assert os.path.exists(out_file)
        with open(out_file, encoding="utf-8") as f:
            content = f.read()

        assert "submit_compile_and_link_jobs" in content
        assert "j_dlc_001" in content
        assert "j_link_002" in content
        assert "j_embed_003" in content
        assert "qnn_context_binary (embedded in ONNX)" in content


class TestPhase3AuthCheck:
    """Tests authentication verification behavior."""

    def test_check_ai_hub_auth_returns_boolean(self) -> None:
        result = check_ai_hub_auth()
        assert isinstance(result, bool)


class TestPhase3CompileWorkflow:
    """Tests the modern submit_compile_and_link_jobs workflow with fallback support."""

    def test_compile_model_qnn_modern_api(self) -> None:
        mock_model = MagicMock()
        mock_cjob = MagicMock(job_id="comp_123")
        mock_ljob = MagicMock(job_id="link_456")
        mock_ejob = MagicMock(job_id="embed_789")
        mock_ejob.get_target_model.return_value = mock_model

        with patch("qai_hub.submit_compile_and_link_jobs") as mock_submit:
            mock_submit.return_value = ([mock_cjob], mock_ljob, mock_ejob)

            target_model, telemetry = compile_model_qnn(
                model_path="detector_quantized.onnx",
                device=MagicMock(),
                name="test_compile",
            )

            assert target_model == mock_model
            assert telemetry["api_used"] == "submit_compile_and_link_jobs"
            assert telemetry["compile_job_id"] == "comp_123"
            assert telemetry["link_job_id"] == "link_456"
            assert telemetry["embed_job_id"] == "embed_789"
            assert telemetry["primary_job_id"] == "embed_789"

            mock_submit.assert_called_once()
            _, kwargs = mock_submit.call_args
            assert kwargs.get("embed_in_onnx") is True

    def test_compile_model_qnn_fallback_on_exception(self) -> None:
        mock_model = MagicMock()
        mock_legacy_job = MagicMock(job_id="legacy_123")
        mock_legacy_job.get_target_model.return_value = mock_model

        with (
            patch(
                "qai_hub.submit_compile_and_link_jobs",
                side_effect=RuntimeError("API not supported on server"),
            ),
            patch("qai_hub.submit_compile_job", return_value=mock_legacy_job) as mock_legacy,
        ):
            target_model, telemetry = compile_model_qnn(
                model_path="detector_quantized.onnx",
                device=MagicMock(),
                name="test_compile",
            )

            assert target_model == mock_model
            assert telemetry["api_used"] == "submit_compile_job"
            assert telemetry["compile_job_id"] == "legacy_123"
            assert telemetry["primary_job_id"] == "legacy_123"
            mock_legacy.assert_called_once()

    def test_compile_model_qnn_force_legacy(self) -> None:
        mock_model = MagicMock()
        mock_legacy_job = MagicMock(job_id="forced_legacy_456")
        mock_legacy_job.get_target_model.return_value = mock_model

        with patch("qai_hub.submit_compile_job", return_value=mock_legacy_job) as mock_legacy:
            target_model, telemetry = compile_model_qnn(
                model_path="detector_quantized.onnx",
                device=MagicMock(),
                name="test_compile",
                force_legacy=True,
            )

            assert target_model == mock_model
            assert telemetry["api_used"] == "submit_compile_job"
            assert telemetry["compile_job_id"] == "forced_legacy_456"
            mock_legacy.assert_called_once()


class TestPhase3MetricExtraction:
    """Tests metric extraction from various Qualcomm AI Hub profile payload structures."""

    def test_extract_with_all_inference_times_and_execution_detail(self) -> None:
        profile_data = {
            "execution_summary": {
                "all_inference_times": [16000, 16100, 15900, 16200, 15800],
                "estimated_inference_time": 15800,
                "estimated_inference_peak_memory": 38809600,
            },
            "execution_detail": [
                {"name": "conv1", "compute_unit": "NPU"},
                {"name": "conv2", "compute_unit": "NPU"},
                {"name": "conv3", "compute_unit": "NPU"},
            ],
        }
        metrics = extract_profile_metrics(profile_data)
        assert metrics["inference_ms"] == 16.0
        assert metrics["npu_cycles_pct"] == 100.0
        assert metrics["cpu_cycles_pct"] == 0.0
        assert round(metrics["peak_memory_mb"], 2) == 37.01

    def test_extract_with_int_estimated_inference_time(self) -> None:
        profile_data = {
            "execution_summary": {
                "estimated_inference_time": 15838,
            },
            "compute_units": {"npu": 100.0, "cpu": 0.0, "gpu": 0.0},
            "memory_metrics": {"peak_memory_bytes": 38809600},
        }
        metrics = extract_profile_metrics(profile_data)
        assert round(metrics["inference_ms"], 2) == 15.84
        assert metrics["npu_cycles_pct"] == 100.0
        assert metrics["cpu_cycles_pct"] == 0.0
        assert round(metrics["peak_memory_mb"], 2) == 37.01

    def test_extract_with_dict_estimated_inference_time(self) -> None:
        profile_data = {
            "execution_summary": {
                "estimated_inference_time": {"median": 16000, "p95": 16500},
            },
            "compute_units": {"npu": 98.5, "cpu": 1.5, "gpu": 0.0},
            "memory_metrics": {"peak_memory_bytes": 31457280},
        }
        metrics = extract_profile_metrics(profile_data)
        assert metrics["inference_ms"] == 16.0
        assert metrics["p95_inference_ms"] == 16.5
        assert metrics["npu_cycles_pct"] == 98.5
        assert metrics["cpu_cycles_pct"] == 1.5
        assert metrics["peak_memory_mb"] == 30.0
