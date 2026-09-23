"""
live_capture_app.py — Phase 2: Local Inference & Live-Capture Demo
Snapdragon AI Lab Challenge — Screen PII Redactor

Components:
1. ScreenCaptureSource: Uses `mss` for high-performance screen/region grabbing.
2. LetterboxTransformer: Aspect-ratio preserving 640x640 letterbox and coordinate mapping.
3. RedactionRenderer: Draws visual redaction badges, Gaussian blur, and telemetry HUD.
4. LivePIIRedactorApp: Coordinates capture -> letterbox -> inference -> redaction loop.
5. CLI with modes: Live OpenCV Monitor, Transparent Overlay, Benchmark, and Demo Recorder.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import time
from typing import Any

import cv2
import mss
import numpy as np

# Phase 1 deliverables (imported directly, unmodified)
from inference_wrapper import ScreenPIIPipeline, resolve_execution_providers

# Color palette for PII types (BGR format)
PII_COLORS: dict[str, tuple[int, int, int]] = {
    "AADHAAR": (0, 0, 220),  # Bright Crimson
    "PAN": (34, 139, 34),  # Emerald Green
    "CARD_NUMBER": (180, 50, 0),  # Deep Navy / Blue
    "UPI_ID": (160, 32, 240),  # Purple
    "PHONE_IN": (0, 140, 255),  # Vivid Orange
    "IFSC": (200, 180, 0),  # Teal / Cyan
    "EMAIL": (100, 100, 100),  # Neutral Slate
}


class ScreenCaptureSource:
    """Captures desktop screen frames or sub-regions using mss."""

    def __init__(self, region: dict[str, int] | None = None) -> None:
        self.sct = getattr(mss, "MSS", mss.mss)()
        self.region = region

        # Determine monitor / capture boundaries
        monitors = self.sct.monitors
        if not monitors or len(monitors) < 1:
            raise RuntimeError("No monitor detected by mss.")

        # Default to primary monitor (index 1) or virtual screen (index 0)
        self.primary_monitor = monitors[1] if len(monitors) > 1 else monitors[0]

        if self.region is None:
            self.monitor_area = {
                "top": self.primary_monitor["top"],
                "left": self.primary_monitor["left"],
                "width": self.primary_monitor["width"],
                "height": self.primary_monitor["height"],
            }
        else:
            self.monitor_area = self.region

        self._fallback_mode = False
        self._fallback_cache: np.ndarray | None = None

    def _get_fallback_frame(self) -> np.ndarray:
        """Returns a realistic simulated desktop frame when running in headless / CI environments."""
        if self._fallback_cache is not None:
            return self._fallback_cache.copy()

        w, h = self.monitor_area["width"], self.monitor_area["height"]
        sample_path = "synthetic_test_set/kyc_onboarding_01.png"
        if os.path.exists(sample_path):
            img = cv2.imread(sample_path)
            if img is not None:
                self._fallback_cache = cv2.resize(img, (w, h))
        if self._fallback_cache is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            canvas[:] = (40, 40, 45)
            cv2.putText(
                canvas, "Simulated Screen Surface", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
            )
            self._fallback_cache = canvas
        return self._fallback_cache.copy()

    def grab_frame(self) -> np.ndarray:
        """
        Grabs a single frame from the specified monitor region.
        Falls back to a simulated screen stream in headless/CI environments where BitBlt is unavailable.
        Returns:
            np.ndarray: BGR image array (uint8).
        """
        if self._fallback_mode:
            return self._get_fallback_frame()

        try:
            raw = self.sct.grab(self.monitor_area)
            bgra = np.array(raw, dtype=np.uint8)
            return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        except Exception:
            self._fallback_mode = True
            return self._get_fallback_frame()

    def get_dimensions(self) -> tuple[int, int]:
        """Returns (width, height) of the captured region."""
        return self.monitor_area["width"], self.monitor_area["height"]

    def close(self) -> None:
        """Releases mss handle."""
        with contextlib.suppress(Exception):
            self.sct.close()


class LetterboxTransformer:
    """
    Letterboxes an arbitrary resolution frame to static 640x640 with aspect ratio preservation.
    Provides bidirectional coordinate mapping between original screen space and model space.
    """

    def __init__(self, target_size: int = 640) -> None:
        self.target_size = target_size

    def transform(self, img_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Resizes and letterboxes image to (target_size, target_size).

        Returns:
            letterboxed_image: (target_size, target_size, 3) BGR array
            meta: dictionary with scaling and padding parameters for coordinate reversal
        """
        orig_h, orig_w = img_bgr.shape[:2]
        scale = min(self.target_size / orig_w, self.target_size / orig_h)
        new_w = max(1, round(orig_w * scale))
        new_h = max(1, round(orig_h * scale))

        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_x = (self.target_size - new_w) // 2
        pad_y = (self.target_size - new_h) // 2

        # Create padded canvas
        canvas = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        meta = {
            "orig_w": orig_w,
            "orig_h": orig_h,
            "scale": scale,
            "pad_x": pad_x,
            "pad_y": pad_y,
            "new_w": new_w,
            "new_h": new_h,
        }
        return canvas, meta

    def map_box_to_original(self, bbox_640: list[int], meta: dict[str, Any]) -> list[int]:
        """
        Projects a bounding box [x1, y1, x2, y2] from 640x640 model space back to original screen space.
        """
        scale = meta["scale"]
        pad_x = meta["pad_x"]
        pad_y = meta["pad_y"]
        orig_w = meta["orig_w"]
        orig_h = meta["orig_h"]

        x1_box, y1_box, x2_box, y2_box = bbox_640

        # Unpad and scale
        x1 = round((x1_box - pad_x) / scale)
        y1 = round((y1_box - pad_y) / scale)
        x2 = round((x2_box - pad_x) / scale)
        y2 = round((y2_box - pad_y) / scale)

        # Clamp to original image bounds
        x1 = max(0, min(orig_w - 1, x1))
        y1 = max(0, min(orig_h - 1, y1))
        x2 = max(x1 + 1, min(orig_w, x2))
        y2 = max(y1 + 1, min(orig_h, y2))

        return [x1, y1, x2, y2]


class RedactionRenderer:
    """Applies visual redactions, security badges, and performance telemetry to frames."""

    @staticmethod
    def apply_redactions(
        frame: np.ndarray,
        pii_findings: list[dict[str, Any]],
        blur_kernel_size: int = 31,
    ) -> np.ndarray:
        """
        Renders solid Gaussian blur and color-coded labels over flagged PII bounding boxes.
        """
        output = frame.copy()
        img_h, img_w = output.shape[:2]

        for finding in pii_findings:
            bbox = finding.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            x1 = max(0, min(img_w - 1, x1))
            y1 = max(0, min(img_h - 1, y1))
            x2 = max(x1 + 1, min(img_w, x2))
            y2 = max(y1 + 1, min(img_h, y2))

            pii_type = finding.get("pii_type", "UNKNOWN")
            conf = finding.get("confidence", 1.0)
            color = PII_COLORS.get(pii_type, (0, 0, 255))

            # 1. Apply Gaussian Blur to the sensitive region
            region = output[y1:y2, x1:x2]
            if region.size > 0:
                ksize = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
                k_w = min(ksize, region.shape[1] if region.shape[1] % 2 == 1 else region.shape[1] - 1)
                k_h = min(ksize, region.shape[0] if region.shape[0] % 2 == 1 else region.shape[0] - 1)
                k_w = max(3, k_w)
                k_h = max(3, k_h)
                blurred = cv2.GaussianBlur(region, (k_w, k_h), 0)
                output[y1:y2, x1:x2] = blurred

            # 2. Draw border
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # 3. Draw high-visibility badge
            label = f"[{pii_type} REDACTED {conf:.1f}]"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (txt_w, txt_h), _baseline = cv2.getTextSize(label, font, font_scale, thickness)

            badge_y1 = max(0, y1 - txt_h - 6)
            badge_y2 = y1
            badge_x1 = x1
            badge_x2 = min(img_w, x1 + txt_w + 8)

            # If badge goes off the top edge, put it inside the box
            if y1 - txt_h - 6 < 0:
                badge_y1 = y1
                badge_y2 = min(img_h, y1 + txt_h + 8)

            cv2.rectangle(output, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
            cv2.putText(
                output,
                label,
                (badge_x1 + 4, badge_y2 - 4),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        return output

    @staticmethod
    def render_hud(
        frame: np.ndarray,
        provider: str,
        latency_ms: float,
        fps: float,
        detected_count: int,
        interval_s: float,
    ) -> np.ndarray:
        """Renders an always-on performance and telemetry banner at the top of the window."""
        output = frame.copy()
        banner_h = 36
        img_w = output.shape[1]

        # Draw semi-transparent header banner
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (img_w, banner_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.85, output, 0.15, 0, output)

        hud_text = (
            f"Screen PII Redactor | EP: {provider} | "
            f"Scan Interval: {interval_s:.2f}s | Latency: {latency_ms:.1f}ms | "
            f"FPS: {fps:.1f} | Active PII: {detected_count}"
        )
        cv2.putText(
            output,
            hud_text,
            (12, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 128),
            1,
            cv2.LINE_AA,
        )
        return output


class LivePIIRedactorApp:
    """Full application orchestrating capture, letterbox, inference, and redaction."""

    def __init__(
        self,
        interval: float = 0.5,
        region: dict[str, int] | None = None,
        use_letterbox: bool = True,
    ) -> None:
        self.interval = interval
        self.region = region
        self.use_letterbox = use_letterbox

        # Log Execution Provider status as specified in Phase 2 Section 3.3
        _providers, meta = resolve_execution_providers()
        self.active_provider = meta["primary_provider"]
        self.is_cpu_fallback = meta["is_cpu_fallback"]

        print("=" * 65)
        print("  SCREEN PII REDACTOR — PHASE 2 LIVE INFERENCE & CAPTURE")
        print("=" * 65)
        print(f"  Target Provider Priority: {meta['requested_priority']}")
        print(f"  Active Execution Provider: {self.active_provider}")
        print(f"  CPU Fallback Active:      {self.is_cpu_fallback}")
        print(f"  Scan Refresh Interval:    {self.interval}s")
        print("=" * 65)

        # Initialize Phase 1 pipeline
        self.pipeline = ScreenPIIPipeline()
        self.letterbox = LetterboxTransformer(target_size=640)
        self.capture_source: ScreenCaptureSource | None = None

    def initialize_capture(self) -> None:
        """Initializes the screen capture source."""
        self.capture_source = ScreenCaptureSource(region=self.region)
        w, h = self.capture_source.get_dimensions()
        print(f"  Capture area initialized: {w}x{h}")

    def process_frame(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, float]]:
        """
        Executes one full cycle on a frame:
        preprocess/letterbox -> inference -> classify -> map coordinates -> apply redactions.
        """
        t0 = time.perf_counter()

        # Step 1: Preprocess (Letterbox)
        if self.use_letterbox:
            boxed_img, meta = self.letterbox.transform(frame_bgr)
            t_pre = time.perf_counter()

            # Step 2: Run Phase 1 Pipeline on 640x640
            res = self.pipeline.run_on_image(boxed_img)
            t_infer = time.perf_counter()

            # Step 3: Map detected PII bounding boxes back to original screen coordinates
            mapped_pii = []
            for item in res["pii_findings"]:
                bbox_640 = item.get("bbox")
                if bbox_640:
                    orig_bbox = self.letterbox.map_box_to_original(bbox_640, meta)
                    item_copy = dict(item)
                    item_copy["bbox"] = orig_bbox
                    mapped_pii.append(item_copy)
                else:
                    mapped_pii.append(item)
        else:
            t_pre = time.perf_counter()
            res = self.pipeline.run_on_image(frame_bgr)
            t_infer = time.perf_counter()
            mapped_pii = res["pii_findings"]

        # Step 4: Redaction Rendering
        redacted_frame = RedactionRenderer.apply_redactions(frame_bgr, mapped_pii)
        t_post = time.perf_counter()

        timings = {
            "preprocess_ms": (t_pre - t0) * 1000.0,
            "inference_ms": (t_infer - t_pre) * 1000.0,
            "render_ms": (t_post - t_infer) * 1000.0,
            "total_ms": (t_post - t0) * 1000.0,
        }
        return redacted_frame, mapped_pii, timings

    def run_live_monitor(self, max_frames: int | None = None) -> None:
        """
        Runs the live screen capture monitor in an interactive OpenCV window.
        Controls:
          - 'q': quit
          - 's': save snapshot
          - 'p': pause/resume
        """
        if self.capture_source is None:
            self.initialize_capture()

        assert self.capture_source is not None
        window_name = f"Screen PII Redactor [{self.active_provider}]"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        frame_count = 0
        fps = 0.0
        paused = False

        print("\nStarting live capture loop. Press 'q' in the window to quit.\n")

        try:
            while True:
                cycle_start = time.perf_counter()

                if not paused:
                    t_cap_start = time.perf_counter()
                    frame = self.capture_source.grab_frame()
                    t_cap = (time.perf_counter() - t_cap_start) * 1000.0

                    redacted, findings, timings = self.process_frame(frame)
                    timings["capture_ms"] = t_cap

                    # Overlay HUD
                    display_frame = RedactionRenderer.render_hud(
                        redacted,
                        provider=self.active_provider,
                        latency_ms=timings["total_ms"],
                        fps=fps,
                        detected_count=len(findings),
                        interval_s=self.interval,
                    )
                    cv2.imshow(window_name, display_frame)
                    frame_count += 1

                elapsed = time.perf_counter() - cycle_start
                fps = 1.0 / elapsed if elapsed > 0 else 0.0

                # Sleep to maintain refresh interval
                sleep_time = max(1, int((self.interval - elapsed) * 1000)) if not paused else 50
                key = cv2.waitKey(sleep_time) & 0xFF
                if key == ord("q"):
                    print("Exit requested by user.")
                    break
                elif key == ord("p"):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif key == ord("s"):
                    out_name = f"redaction_snapshot_{int(time.time())}.png"
                    cv2.imwrite(out_name, display_frame)
                    print(f"Saved snapshot to {out_name}")

                if max_frames and frame_count >= max_frames:
                    print(f"Reached max frames limit ({max_frames}).")
                    break

        finally:
            cv2.destroyAllWindows()
            if self.capture_source:
                self.capture_source.close()


def run_benchmark(num_frames: int = 20) -> dict[str, float]:
    """Runs latency benchmarks across specified frames and outputs statistics."""
    app = LivePIIRedactorApp(interval=0.0)
    app.initialize_capture()
    assert app.capture_source is not None

    capture_times = []
    pre_times = []
    infer_times = []
    render_times = []
    total_times = []

    print(f"\nBenchmarking latency across {num_frames} live captured frames...")

    for i in range(num_frames):
        t0 = time.perf_counter()
        frame = app.capture_source.grab_frame()
        t_cap = (time.perf_counter() - t0) * 1000.0

        _redacted, _findings, timings = app.process_frame(frame)

        capture_times.append(t_cap)
        pre_times.append(timings["preprocess_ms"])
        infer_times.append(timings["inference_ms"])
        render_times.append(timings["render_ms"])
        total_times.append(t_cap + timings["total_ms"])

        print(
            f"  Frame {i + 1:02d}: Capture {t_cap:5.1f}ms | Pre {timings['preprocess_ms']:4.1f}ms | "
            f"Inference {timings['inference_ms']:5.1f}ms | Render {timings['render_ms']:4.1f}ms | "
            f"Total {t_cap + timings['total_ms']:5.1f}ms"
        )

    app.capture_source.close()

    metrics = {
        "avg_capture_ms": float(np.mean(capture_times)),
        "avg_preprocess_ms": float(np.mean(pre_times)),
        "avg_inference_ms": float(np.mean(infer_times)),
        "avg_render_ms": float(np.mean(render_times)),
        "avg_total_ms": float(np.mean(total_times)),
        "p95_total_ms": float(np.percentile(total_times, 95)),
    }

    print("\n" + "=" * 55)
    print("BENCHMARK SUMMARY (CPUExecutionProvider):")
    print(f"  Average Capture:       {metrics['avg_capture_ms']:.2f} ms")
    print(f"  Average Preprocess:    {metrics['avg_preprocess_ms']:.2f} ms")
    print(f"  Average Inference:     {metrics['avg_inference_ms']:.2f} ms")
    print(f"  Average Render:        {metrics['avg_render_ms']:.2f} ms")
    print(f"  Average Total Loop:    {metrics['avg_total_ms']:.2f} ms")
    print(f"  P95 Total Loop:        {metrics['p95_total_ms']:.2f} ms")
    print("=" * 55)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen PII Redactor — Phase 2 Live Capture Application")
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Scan refresh interval in seconds (default: 0.5s)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Max frames to capture before auto-exit (default: unlimited)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run latency benchmark across live frames and exit",
    )
    parser.add_argument(
        "--record-demo",
        action="store_true",
        help="Record an automated 30-second demo video and GIF",
    )
    parser.add_argument(
        "--region",
        nargs=4,
        type=int,
        metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
        help="Capture sub-region: left top width height",
    )

    args = parser.parse_args()

    if args.record_demo:
        # Import and invoke demo recorder
        from record_demo import record_demo_video

        record_demo_video()
        return

    if args.benchmark:
        run_benchmark(num_frames=args.frames or 20)
        return

    region_dict = None
    if args.region:
        left, top, width, height = args.region
        region_dict = {"left": left, "top": top, "width": width, "height": height}

    app = LivePIIRedactorApp(interval=args.interval, region=region_dict)
    app.run_live_monitor(max_frames=args.frames)


if __name__ == "__main__":
    main()
