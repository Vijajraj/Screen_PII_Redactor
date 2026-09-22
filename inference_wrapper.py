"""
inference_wrapper.py — Cross-Device Execution Provider Abstraction & Inference Pipeline
Phase 1 Deliverable — Screen PII Redactor (Snapdragon AI Lab Challenge)

Section 3.2 Specification:
- Hardware-agnostic abstraction layer.
- Priority order:
    1. QNNExecutionProvider        (Snapdragon laptops / Hexagon NPU - required submission target)
    2. NNAPIExecutionProvider      (Android phones - Phone NPU/DSP)
    3. CoreMLExecutionProvider      (iOS / Mac - Apple Neural Engine)
    4. CPUExecutionProvider         (Universal fallback)
- Confirms CPUExecutionProvider fallback in local (non-Snapdragon) environment.
"""

import os
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

# Import the portable PII classifier deliverable
from pii_classifier import classify_text

# =====================================================================
# 1. EXECUTION PROVIDER ABSTRACTION LAYER (Section 3.2)
# =====================================================================

TARGET_PROVIDER_PRIORITY = [
    "QNNExecutionProvider",  # Snapdragon Hexagon NPU (Submission Target)
    "NNAPIExecutionProvider",  # Android NPU / DSP
    "CoreMLExecutionProvider",  # Apple Neural Engine
    "CPUExecutionProvider",  # Universal Fallback
]


def resolve_execution_providers(preferred_providers: list[str] | None = None) -> tuple[list[str], dict[str, Any]]:
    """
    Selects available ONNX Runtime execution providers in priority order.
    Returns:
      (active_providers_list, metadata_dict)
    """
    candidate_priority = preferred_providers or TARGET_PROVIDER_PRIORITY
    available = set(ort.get_available_providers())

    selected = [p for p in candidate_priority if p in available]
    if not selected:
        selected = ["CPUExecutionProvider"]

    metadata = {
        "all_installed_providers": list(available),
        "requested_priority": candidate_priority,
        "selected_providers": selected,
        "primary_provider": selected[0],
        "is_npu_accelerated": any(
            p in ("QNNExecutionProvider", "NNAPIExecutionProvider", "CoreMLExecutionProvider") for p in selected
        ),
        "is_cpu_fallback": (selected == ["CPUExecutionProvider"]),
    }
    return selected, metadata


# =====================================================================
# 2. DBNet TEXT DETECTION (PP-OCRv4 Mobile Detection, Static 640x640)
# =====================================================================


class PPOCRv4Detector:
    """
    PP-OCRv4 DBNet Mobile Text Detector.
    Fixed static input shape: [1, 3, 640, 640]
    """

    def __init__(
        self,
        model_path: str = "detector_quantized.onnx",
        providers: list[str] | None = None,
        thresh: float = 0.3,
        box_thresh: float = 0.5,
        unclip_ratio: float = 1.5,
        max_candidates: int = 1000,
    ):
        self.model_path = model_path
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio
        self.max_candidates = max_candidates

        self.providers, self.provider_metadata = resolve_execution_providers(providers)

        # Session options for optimal multi-threaded inference
        session_opts = ort.SessionOptions()
        session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(self.model_path, sess_options=session_opts, providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, img_bgr: np.ndarray) -> tuple[np.ndarray, float, float]:
        """
        Preprocesses image to static 640x640 with standard DBNet normalization.
        Returns: (tensor_nchw, scale_x, scale_y)
        """
        orig_h, orig_w = img_bgr.shape[:2]
        target_w, target_h = 640, 640

        resized = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        # Normalize: (img / 255.0 - mean) / std
        img_float = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_img = (img_float - mean) / std

        # HWC -> CHW -> NCHW
        tensor = np.transpose(norm_img, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        scale_x = orig_w / float(target_w)
        scale_y = orig_h / float(target_h)
        return tensor, scale_x, scale_y

    def postprocess(
        self, pred_map: np.ndarray, scale_x: float, scale_y: float, orig_w: int, orig_h: int
    ) -> list[dict[str, Any]]:
        """
        Converts probability heatmap into bounding boxes [x1, y1, x2, y2].
        """
        # pred_map shape: [1, 1, 640, 640] -> squeeze to [640, 640]
        prob_map = np.squeeze(pred_map)
        binary_mask = (prob_map > self.thresh).astype(np.uint8)

        contours, _ = cv2.findContours(binary_mask * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        boxes: list[dict[str, Any]] = []
        for c in contours[: self.max_candidates]:
            if cv2.contourArea(c) < 16:
                continue

            x, y, w, h = cv2.boundingRect(c)
            # Unclip / expand slightly to cover word edges
            expand_w = int(w * (self.unclip_ratio - 1.0) / 2)
            expand_h = int(h * (self.unclip_ratio - 1.0) / 2)

            x1_640 = max(0, x - expand_w)
            y1_640 = max(0, y - expand_h)
            x2_640 = min(640, x + w + expand_w)
            y2_640 = min(640, y + h + expand_h)

            # Confidence is average probability within the detected region
            region_prob = prob_map[y : y + h, x : x + w]
            score = float(region_prob.mean()) if region_prob.size > 0 else 0.0

            if score < self.box_thresh:
                continue

            # Scale coordinates back to original image
            x1 = round(x1_640 * scale_x)
            y1 = round(y1_640 * scale_y)
            x2 = round(x2_640 * scale_x)
            y2 = round(y2_640 * scale_y)

            # Clamp to image dimensions
            x1 = max(0, min(orig_w - 1, x1))
            y1 = max(0, min(orig_h - 1, y1))
            x2 = max(x1 + 1, min(orig_w, x2))
            y2 = max(y1 + 1, min(orig_h, y2))

            boxes.append({"bbox": [x1, y1, x2, y2], "det_confidence": round(score, 4)})

        # Sort boxes top-to-bottom, left-to-right
        boxes.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
        return boxes

    def detect(self, img_bgr: np.ndarray) -> list[dict[str, Any]]:
        """Run text detection on BGR image."""
        orig_h, orig_w = img_bgr.shape[:2]
        tensor, scale_x, scale_y = self.preprocess(img_bgr)
        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        pred_map = outputs[0]
        return self.postprocess(pred_map, scale_x, scale_y, orig_w, orig_h)


# =====================================================================
# 3. TEXT RECOGNITION (SVTR-based PP-OCRv4 Mobile Recognizer)
# =====================================================================


class PPOCRv4Recognizer:
    """
    PP-OCRv4 SVTR Text Recognizer with CTC Greedy Decoding.
    """

    def __init__(
        self,
        model_path: str = "models/rec/english/model.onnx",
        dict_path: str = "models/rec/english/dict.txt",
        providers: list[str] | None = None,
    ):
        self.model_path = model_path
        self.providers, _ = resolve_execution_providers(providers)

        # Load dictionary
        if os.path.exists(dict_path):
            with open(dict_path, encoding="utf-8") as f:
                self.character = [line.strip("\r\n") for line in f.readlines()]
        else:
            self.character = []

        if os.path.exists(model_path):
            self.session = ort.InferenceSession(self.model_path, providers=self.providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        else:
            self.session = None

    def recognize_crop(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        """Recognize cropped text image."""
        if self.session is None or crop_bgr.size == 0 or not self.character:
            return "", 0.0

        h, w = crop_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return "", 0.0
        target_h = 48
        target_w = max(round(w * target_h / float(h)), 16)
        target_w = min(target_w, 640)

        resized = cv2.resize(crop_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img_float = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        tensor = np.transpose(img_float, (2, 0, 1))[np.newaxis, :]

        preds = self.session.run([self.output_name], {self.input_name: tensor})[0]
        # CTC Greedy decode: index 0 is blank; indices >= 1 map to self.character[idx]
        pred_indices = np.argmax(preds, axis=2)[0]
        pred_probs = np.max(preds, axis=2)[0]

        char_list = []
        conf_list = []
        prev_idx = -1
        for idx, prob in zip(pred_indices, pred_probs, strict=False):
            if idx != 0 and idx != prev_idx and idx < len(self.character):
                char_list.append(self.character[idx])
                conf_list.append(float(prob))
            prev_idx = idx

        text = "".join(char_list).strip()
        confidence = float(np.mean(conf_list)) if conf_list else 0.0
        return text, confidence


# =====================================================================
# 4. END-TO-END PIPELINE (Detection + Recognition + PII Classification)
# =====================================================================


class ScreenPIIPipeline:
    """
    Complete Pipeline:
      1. Detect text regions using quantized static ONNX DBNet model
      2. Recognize text string per region
      3. Classify sensitive Indian & Universal PII via pii_classifier
    """

    def __init__(
        self,
        detector_path: str = "detector_quantized.onnx",
        rec_model_path: str = "models/rec/english/model.onnx",
        rec_dict_path: str = "models/rec/english/dict.txt",
        providers: list[str] | None = None,
    ):
        self.detector = PPOCRv4Detector(model_path=detector_path, providers=providers)
        self.recognizer = PPOCRv4Recognizer(model_path=rec_model_path, dict_path=rec_dict_path, providers=providers)
        self.execution_provider = self.detector.provider_metadata["primary_provider"]

    def run_on_image(self, image_input: str | np.ndarray | Image.Image) -> dict[str, Any]:
        """
        Executes end-to-end pipeline on image input.
        Returns:
          {
            "execution_provider": str,
            "detected_text_regions": [ {bbox, text, det_confidence, rec_confidence} ],
            "pii_findings": [ {bbox, pii_type, matched_text, confidence} ]
          }
        """
        # Load image to BGR numpy array
        if isinstance(image_input, str):
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                raise ValueError(f"Could not load image from path: {image_input}")
        elif isinstance(image_input, Image.Image):
            img_rgb = np.array(image_input.convert("RGB"))
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input
        else:
            raise TypeError("Unsupported image input type")

        # Step 1: Detect text bounding boxes
        detected_boxes = self.detector.detect(img_bgr)

        # Step 2: Recognize text for each bounding box
        text_regions = []
        all_pii_matches = []

        for item in detected_boxes:
            x1, y1, x2, y2 = item["bbox"]
            crop = img_bgr[y1:y2, x1:x2]

            rec_text, rec_conf = self.recognizer.recognize_crop(crop)
            region_record = {
                "bbox": [x1, y1, x2, y2],
                "text": rec_text,
                "det_confidence": item["det_confidence"],
                "rec_confidence": round(rec_conf, 4),
            }
            text_regions.append(region_record)

            # Step 3: Run PII classification on recognized text
            if rec_text:
                pii_results = classify_text(rec_text, bbox=[x1, y1, x2, y2])
                all_pii_matches.extend(pii_results)

        return {
            "execution_provider": self.execution_provider,
            "total_text_regions": len(text_regions),
            "detected_text_regions": text_regions,
            "pii_findings": all_pii_matches,
        }


# Self-test when executed directly
if __name__ == "__main__":
    providers, meta = resolve_execution_providers()
    print("=" * 60)
    print("EP-ABSTRACTION INFERENCE WRAPPER SELF-TEST")
    print("=" * 60)
    print(f"Target Priority Order: {meta['requested_priority']}")
    print(f"Installed ORT Providers: {meta['all_installed_providers']}")
    print(f"Selected Provider(s):   {meta['selected_providers']}")
    print(f"Primary Active Provider: {meta['primary_provider']}")
    print(f"Is CPU Fallback Active: {meta['is_cpu_fallback']}")
    print("=" * 60)
