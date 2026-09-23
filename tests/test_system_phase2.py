"""
test_system_phase2.py — System-Level End-to-End Tests for Phase 2 CLI Scripts & Artifacts
Snapdragon AI Lab Challenge — Screen PII Redactor

Tests:
1. live_capture_app.py CLI options (--help, --benchmark) executed via subprocess.
2. evaluate_phase2.py execution and markdown report generation.
3. record_demo.py execution and video/GIF stream integrity.
4. Output artifacts structure and metadata validation.
"""

from __future__ import annotations

import os
import subprocess
import sys

import cv2
from PIL import Image


class TestPhase2CLISystem:
    """System-level tests executing Phase 2 CLI tools via subprocess."""

    def test_live_capture_app_help(self) -> None:
        cmd = [sys.executable, "live_capture_app.py", "--help"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert res.returncode == 0
        assert "Screen PII Redactor" in res.stdout
        assert "--interval" in res.stdout
        assert "--benchmark" in res.stdout
        assert "--record-demo" in res.stdout

    def test_live_capture_app_benchmark_mode(self) -> None:
        cmd = [sys.executable, "live_capture_app.py", "--benchmark", "--frames", "3"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        assert res.returncode == 0
        assert "BENCHMARK SUMMARY" in res.stdout
        assert "Average Total Loop:" in res.stdout

    def test_evaluate_phase2_execution(self) -> None:
        cmd = [sys.executable, "evaluate_phase2.py"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        assert res.returncode == 0
        assert "PHASE 2 EVALUATION COMPLETE" in res.stdout
        assert os.path.exists("phase2_results.md")


class TestPhase2ArtifactIntegrity:
    """Verifies that generated video, GIF, and screenshot artifacts are structurally valid."""

    def test_demo_video_mp4_integrity(self) -> None:
        video_path = "demo_video.mp4"
        assert os.path.exists(video_path)

        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened()

        ret, frame = cap.read()
        assert ret is True
        assert frame is not None
        assert frame.shape == (640, 1024, 3)
        cap.release()

    def test_demo_video_gif_integrity(self) -> None:
        gif_path = "demo_video.gif"
        assert os.path.exists(gif_path)

        with Image.open(gif_path) as img:
            assert getattr(img, "is_animated", False) is True
            assert getattr(img, "n_frames", 1) >= 10
            assert img.size == (640, 400)

    def test_results_screenshots_integrity(self) -> None:
        sample_path = "results/phase2_redaction_sample.png"
        clean_path = "results/phase2_clean_control.png"

        assert os.path.exists(sample_path)
        assert os.path.exists(clean_path)

        img_sample = cv2.imread(sample_path)
        img_clean = cv2.imread(clean_path)

        assert img_sample is not None and img_sample.size > 0
        assert img_clean is not None and img_clean.size > 0
