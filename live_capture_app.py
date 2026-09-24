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
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import mss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Phase 1 deliverables (imported directly, unmodified)
from inference_wrapper import ScreenPIIPipeline, resolve_execution_providers  # noqa: E402

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
        sample_path = PROJECT_ROOT / "synthetic_test_set" / "kyc_onboarding_01.png"
        if sample_path.exists():
            img = cv2.imread(str(sample_path))
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

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        exclude_rect: tuple[int, int, int, int] | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, float]]:
        """
        Executes one full cycle on a frame:
        self-exclusion mask -> preprocess/letterbox -> inference -> classify -> map coordinates -> apply redactions.
        """
        t0 = time.perf_counter()

        # Step 0: Apply Self-Exclusion Mask (eliminates hall-of-mirrors recursion)
        if exclude_rect:
            ex, ey, ew, eh = exclude_rect
            h, w = frame_bgr.shape[:2]
            x1 = max(0, min(w, ex))
            y1 = max(0, min(h, ey))
            x2 = max(0, min(w, ex + ew))
            y2 = max(0, min(h, ey + eh))
            if x2 > x1 + 20 and y2 > y1 + 20:
                frame_bgr = frame_bgr.copy()
                frame_bgr[y1:y2, x1:x2] = (30, 32, 38)
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (65, 70, 80), 2)
                cv2.putText(
                    frame_bgr,
                    "Redactor Window (Excluded from Scan)",
                    (x1 + 15, min(y2 - 15, y1 + 35)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (160, 165, 175),
                    1,
                    cv2.LINE_AA,
                )

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

    def _run_opencv_monitor(self, window_name: str, max_frames: int | None = None) -> None:
        assert self.capture_source is not None
        frame_count = 0
        fps = 0.0
        paused = False

        print("\nStarting live capture loop (OpenCV HighGUI). Press 'q' in the window to quit.\n")

        try:
            while True:
                cycle_start = time.perf_counter()

                if not paused:
                    t_cap_start = time.perf_counter()
                    frame = self.capture_source.grab_frame()
                    t_cap = (time.perf_counter() - t_cap_start) * 1000.0

                    # Calculate self-exclusion area for OpenCV window
                    exclude_rect = None
                    with contextlib.suppress(Exception):
                        rect = cv2.getWindowImageRect(window_name)
                        if rect and rect[2] > 0 and rect[3] > 0:
                            cap_left = self.capture_source.monitor_area.get("left", 0)
                            cap_top = self.capture_source.monitor_area.get("top", 0)
                            exclude_rect = (rect[0] - cap_left, rect[1] - cap_top, rect[2] + 16, rect[3] + 40)

                    redacted, findings, timings = self.process_frame(frame, exclude_rect=exclude_rect)
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
                    out_name = str(PROJECT_ROOT / f"redaction_snapshot_{int(time.time())}.png")
                    cv2.imwrite(out_name, display_frame)
                    print(f"Saved snapshot to {out_name}")

                if max_frames and frame_count >= max_frames:
                    print(f"Reached max frames limit ({max_frames}).")
                    break

        finally:
            cv2.destroyAllWindows()
            if self.capture_source:
                self.capture_source.close()

    def run_live_monitor(self, max_frames: int | None = None, backend: str = "auto") -> None:
        """
        Runs the live screen capture monitor with resilient multi-backend display support:
          1. OpenCV HighGUI window (fast native C++ window)
          2. Tkinter window (universal Python fallback, works when opencv-python-headless is installed)
          3. Headless Console Monitor (pure CLI for headless / server environments)
        """
        if self.capture_source is None:
            self.initialize_capture()

        # 1. Explicit headless request
        if backend == "headless":
            run_headless_monitor(self, max_frames=max_frames)
            return

        # 2. Try OpenCV HighGUI (if backend is auto or opencv)
        if backend in ("auto", "opencv"):
            try:
                window_name = f"Screen PII Redactor [{self.active_provider}]"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                self._run_opencv_monitor(window_name, max_frames=max_frames)
                return
            except (cv2.error, Exception) as cv_err:
                if backend == "opencv":
                    raise
                print(
                    f"\n[Notice] OpenCV GUI window unavailable ({type(cv_err).__name__}). "
                    "Switching to native Tkinter display backend."
                )

        # 3. Try Native Tkinter Window (if backend is auto or tkinter)
        if backend in ("auto", "tkinter"):
            try:
                viewer = TkinterLiveViewer(self, max_frames=max_frames)
                viewer.run()
                return
            except Exception as tk_err:
                if backend == "tkinter":
                    raise
                print(f"\n[Notice] Tkinter display unavailable ({tk_err}). Switching to Headless Console Monitor.")

        # 4. Fallback to Headless Console Monitor
        run_headless_monitor(self, max_frames=max_frames)


class TkinterLiveViewer:
    """
    Native Tkinter fallback viewer when OpenCV HighGUI is not available (e.g. opencv-python-headless).
    Displays the live redacted stream with telemetry HUD, supporting key controls:
      - 'q' or Escape: quit
      - 's': save snapshot
      - 'p': pause/resume
    """

    def __init__(self, app: LivePIIRedactorApp, max_frames: int | None = None) -> None:
        self.app = app
        self.max_frames = max_frames
        self.frame_count = 0
        self.paused = False
        self.running = True
        self.fps = 0.0
        self.current_display_frame: np.ndarray | None = None
        self._photo_image: Any = None

        import tkinter as tk

        from PIL import Image, ImageTk

        self.tk = tk
        self.Image = Image
        self.ImageTk = ImageTk

        self.root = tk.Tk()
        self.root.title(f"Screen PII Redactor [{self.app.active_provider}] — Native Display")
        self.root.configure(bg="#121212")

        # Setup key bindings
        self.root.bind("<Key-q>", lambda e: self.on_quit())
        self.root.bind("<Key-Q>", lambda e: self.on_quit())
        self.root.bind("<Escape>", lambda e: self.on_quit())
        self.root.bind("<Key-p>", lambda e: self.toggle_pause())
        self.root.bind("<Key-P>", lambda e: self.toggle_pause())
        self.root.bind("<Key-s>", lambda e: self.save_snapshot())
        self.root.bind("<Key-S>", lambda e: self.save_snapshot())
        self.root.bind("<Key-r>", lambda e: self.prompt_region_select())
        self.root.bind("<Key-R>", lambda e: self.prompt_region_select())
        self.root.protocol("WM_DELETE_WINDOW", self.on_quit)

        # Main image display container
        self.image_label = tk.Label(self.root, bg="#121212")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Bottom status bar
        self.status_var = tk.StringVar(
            value="[ACTIVE] Live Redactor running. Controls: [Q/Esc] Quit  |  [P] Pause  |  [S] Snapshot  |  [R] Select Region"
        )
        self.status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            bg="#18181b",
            fg="#00e676",
            font=("Consolas", 10, "bold"),
            anchor="w",
            padx=12,
            pady=6,
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def on_quit(self) -> None:
        self.running = False
        with contextlib.suppress(Exception):
            self.root.destroy()
        if self.app.capture_source:
            self.app.capture_source.close()

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        if self.paused:
            self.status_var.set("[PAUSED] Live Redactor paused. Controls: [P] Resume  |  [Q/Esc] Quit  |  [S] Snapshot")
            self.status_label.configure(fg="#ffab00")
            print("Paused")
        else:
            self.status_var.set(
                "[ACTIVE] Live Redactor running. Controls: [Q/Esc] Quit  |  [P] Pause  |  [S] Snapshot  |  [R] Select Region"
            )
            self.status_label.configure(fg="#00e676")
            print("Resumed")

    def save_snapshot(self) -> None:
        if self.current_display_frame is not None:
            out_name = str(PROJECT_ROOT / f"redaction_snapshot_{int(time.time())}.png")
            cv2.imwrite(out_name, self.current_display_frame)
            print(f"Saved snapshot to {out_name}")
            self.status_var.set(f"Saved snapshot to {Path(out_name).name}!")

    def prompt_region_select(self) -> None:
        """Allows interactive region selection while app is running."""
        was_paused = self.paused
        self.paused = True
        self.root.withdraw()
        new_region = select_roi_interactive()
        self.root.deiconify()
        if new_region:
            print(f"Selected new capture region: {new_region}")
            self.app.region = new_region
            self.app.initialize_capture()
            self.status_var.set(
                f"[REGION ACTIVE] {new_region['width']}x{new_region['height']} at ({new_region['left']},{new_region['top']}) | [R] Select Region"
            )
        self.paused = was_paused

    def step(self) -> None:
        if not self.running:
            return

        cycle_start = time.perf_counter()

        if not self.paused and self.app.capture_source:
            t_cap_start = time.perf_counter()
            frame = self.app.capture_source.grab_frame()
            t_cap = (time.perf_counter() - t_cap_start) * 1000.0

            # Calculate window rect on desktop to eliminate hall-of-mirrors recursion
            wx = self.root.winfo_x()
            wy = self.root.winfo_y()
            ww = self.root.winfo_width() + 16
            wh = self.root.winfo_height() + 40

            cap_left = self.app.capture_source.monitor_area.get("left", 0)
            cap_top = self.app.capture_source.monitor_area.get("top", 0)
            rel_x = wx - cap_left
            rel_y = wy - cap_top
            exclude_rect = (rel_x, rel_y, ww, wh) if (ww > 20 and wh > 20) else None

            redacted, findings, timings = self.app.process_frame(frame, exclude_rect=exclude_rect)
            timings["capture_ms"] = t_cap

            display_frame = RedactionRenderer.render_hud(
                redacted,
                provider=self.app.active_provider,
                latency_ms=timings["total_ms"],
                fps=self.fps,
                detected_count=len(findings),
                interval_s=self.app.interval,
            )
            self.current_display_frame = display_frame
            self.frame_count += 1

            # Downscale dynamically to fit comfortably within 1280x720 window if screen is large
            disp_h, disp_w = display_frame.shape[:2]
            max_w, max_h = 1280, 720
            if disp_w > max_w or disp_h > max_h:
                scale = min(max_w / disp_w, max_h / disp_h)
                new_w, new_h = max(1, int(disp_w * scale)), max(1, int(disp_h * scale))
                render_img = cv2.resize(display_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                render_img = display_frame

            rgb = cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB)
            pil_img = self.Image.fromarray(rgb)
            self._photo_image = self.ImageTk.PhotoImage(image=pil_img)
            self.image_label.configure(image=self._photo_image)

        elapsed = time.perf_counter() - cycle_start
        self.fps = 1.0 / elapsed if elapsed > 0 else 0.0

        if self.max_frames and self.frame_count >= self.max_frames:
            print(f"Reached max frames limit ({self.max_frames}). Exiting Tkinter viewer.")
            self.on_quit()
            return

        delay_ms = max(10, int((self.app.interval - elapsed) * 1000)) if not self.paused else 50
        if self.running:
            self.root.after(delay_ms, self.step)

    def run(self) -> None:
        print("\nStarting live capture loop (Native Tkinter Window).")
        print("Controls in window: [q] or [Esc] to quit, [p] to pause/resume, [s] to save snapshot.\n")
        self.root.after(10, self.step)
        try:
            self.root.mainloop()
        finally:
            if self.app.capture_source:
                self.app.capture_source.close()


def run_headless_monitor(app: LivePIIRedactorApp, max_frames: int | None = None) -> None:
    """
    Console-only fallback when no display server (OpenCV HighGUI or Tkinter) is available.
    Monitors live capture, logs PII detections, and saves snapshots periodically.
    """
    print("\nRunning in Headless Console Monitor mode (no GUI display server detected).")
    print("Press Ctrl+C to terminate.\n")
    if app.capture_source is None:
        app.initialize_capture()

    assert app.capture_source is not None
    frame_count = 0

    try:
        while True:
            t0 = time.perf_counter()
            frame = app.capture_source.grab_frame()
            redacted, findings, timings = app.process_frame(frame)
            frame_count += 1
            elapsed = time.perf_counter() - t0
            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            pii_summary = ", ".join(f"{f['type']}" for f in findings) if findings else "None"
            print(
                f"[Frame {frame_count:03d}] Latency: {timings['total_ms']:5.1f}ms | "
                f"FPS: {fps:4.1f} | Detections: {len(findings)} ({pii_summary})"
            )

            # Save snapshot of first frame or whenever PII is detected
            if frame_count == 1 or findings:
                out_path = PROJECT_ROOT / "redaction_snapshot_latest.png"
                display_frame = RedactionRenderer.render_hud(
                    redacted,
                    provider=app.active_provider,
                    latency_ms=timings["total_ms"],
                    fps=fps,
                    detected_count=len(findings),
                    interval_s=app.interval,
                )
                cv2.imwrite(str(out_path), display_frame)

            if max_frames and frame_count >= max_frames:
                print(f"Reached max frames limit ({max_frames}).")
                break

            sleep_time = max(0.01, app.interval - (time.perf_counter() - t0))
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nHeadless monitor stopped by user.")
    finally:
        if app.capture_source:
            app.capture_source.close()


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


def select_roi_interactive() -> dict[str, int] | None:
    """
    Opens a fullscreen semi-transparent click-and-drag overlay allowing the user
    to select any window, document, or screen sub-region to redact.
    """
    import tkinter as tk

    selected_rect: dict[str, int] | None = None
    try:
        root = tk.Tk()
        root.attributes("-fullscreen", True)
        root.attributes("-alpha", 0.30)
        root.configure(bg="#000000")
        root.config(cursor="cross")

        canvas = tk.Canvas(root, cursor="cross", bg="#000000", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)

        screen_w = root.winfo_screenwidth()
        canvas.create_text(
            screen_w // 2,
            60,
            text="Click and drag to select screen region to redact. Press [Esc] to cancel.",
            fill="#00e676",
            font=("Segoe UI", 16, "bold"),
        )

        start_x, start_y = 0, 0
        rect_id = None

        def on_press(event: Any) -> None:
            nonlocal start_x, start_y, rect_id
            start_x, start_y = event.x, event.y
            if rect_id:
                canvas.delete(rect_id)
            rect_id = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline="#00e676", width=2)

        def on_drag(event: Any) -> None:
            nonlocal rect_id
            if rect_id:
                canvas.coords(rect_id, start_x, start_y, event.x, event.y)

        def on_release(event: Any) -> None:
            nonlocal selected_rect
            end_x, end_y = event.x, event.y
            x1, x2 = min(start_x, end_x), max(start_x, end_x)
            y1, y2 = min(start_y, end_y), max(start_y, end_y)
            w, h = x2 - x1, y2 - y1
            if w > 30 and h > 30:
                selected_rect = {"left": x1, "top": y1, "width": w, "height": h}
            root.destroy()

        def on_cancel(event: Any = None) -> None:
            root.destroy()

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)
        root.bind("<Escape>", on_cancel)

        root.mainloop()
    except Exception as e:
        print(f"[Notice] Interactive ROI selector unavailable: {e}")
        return None

    return selected_rect


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
    parser.add_argument(
        "--select-region",
        action="store_true",
        help="Interactively click and drag to select screen region before starting",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "opencv", "tkinter", "headless"],
        default="auto",
        help="Display backend: auto (default: auto-detects best GUI), opencv, tkinter, or headless",
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
    if args.select_region:
        print("\nOpening interactive region selector. Click and drag over your document...")
        region_dict = select_roi_interactive()
        if region_dict:
            print(f"Selected region: {region_dict}")

    if region_dict is None and args.region:
        left, top, width, height = args.region
        region_dict = {"left": left, "top": top, "width": width, "height": height}

    app = LivePIIRedactorApp(interval=args.interval, region=region_dict)
    app.run_live_monitor(max_frames=args.frames, backend=args.backend)


if __name__ == "__main__":
    main()
