# Screen PII Redactor 🛡️

**On-Device Sensitive PII Detection & Redaction for Indian & Universal Identifiers**  
*Snapdragon AI Lab Challenge — Phase 1: Model Build & Export (Local)*

[![Tests](https://img.shields.io/badge/tests-13%20passed-brightgreen.svg)]()
[![Model](https://img.shields.io/badge/ONNX-Static%20640x640-blue.svg)]()
[![Quantization](https://img.shields.io/badge/INT8-1.27%20MB%20(3.57x)-orange.svg)]()
[![Target EP](https://img.shields.io/badge/Qualcomm-QNN%20NPU-purple.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Overview

**Screen PII Redactor** is a high-performance, on-device privacy engine built to detect and redact sensitive Personally Identifiable Information (PII) from screen captures and desktop frames. Designed specifically for **Qualcomm Snapdragon X-Elite / Hexagon NPU** hardware via the **QNN Execution Provider**, the architecture enforces static tensor shapes, INT8 quantization, and deterministic checksum-backed classification.

### Pipeline Architecture

```
Input Image / Screenshot Frame
    │
    ▼
[1] Text Detection (PP-OCRv4 Mobile DBNet — INT8 Static ONNX [1, 3, 640, 640])
    │ ──► Extracts text regions & bounding boxes [x1, y1, x2, y2]
    ▼
[2] Text Recognition (SVTR-based Mobile Recognizer with CTC Greedy Decoding)
    │ ──► Extracts raw text strings per region
    ▼
[3] PII Classification Layer (Deterministic Regex + Mathematical Checksums)
    │ ──► Validates Aadhaar (Verhoeff), Cards (Luhn), PAN, UPI, Phone, IFSC, Email
    ▼
Output: List of {bbox: [x1, y1, x2, y2], pii_type: str, matched_text: str, confidence: float}
```

---

## 2. Phase 1 Deliverables (Handoff to Phase 2)

| # | Deliverable File | Description | Status |
|:---:|---|---|:---:|
| 1 | [`detector_quantized.onnx`](detector_quantized.onnx) | Final INT8 quantized DBNet text detector (Fixed static shape: `[1, 3, 640, 640]`, 1.27 MB) | ✅ Complete |
| 2 | [`pii_classifier.py`](pii_classifier.py) | Standalone, portable PII classifier (pure Python standard library + `re` only, zero external dependencies) | ✅ Complete |
| 3 | [`inference_wrapper.py`](inference_wrapper.py) | Hardware-agnostic EP-abstraction inference engine with QNN/NNAPI/CoreML/CPU priority fallback | ✅ Complete |
| 4 | [`synthetic_test_set/`](synthetic_test_set/) | 20 synthetic mock screenshots across 5 categories + `ground_truth.json` catalog | ✅ Complete |
| 5 | [`phase1_results.md`](phase1_results.md) | Measured validation report with real benchmarks (model compression, MAE drift, precision/recall) | ✅ Complete |
| 6 | [`phase1_notebook.ipynb`](phase1_notebook.ipynb) | Complete 10-section Jupyter notebook demonstrating the entire pipeline | ✅ Complete |

---

## 3. Execution Provider Abstraction (Cross-Device Portability)

The pipeline is hardware-agnostic; only the **ONNX Runtime execution provider** changes per target device:

| Platform | Execution Provider | Target Hardware | Fallback Priority |
|---|---|---|:---:|
| **Snapdragon Laptops (Windows)** | `QNNExecutionProvider` | **Qualcomm Hexagon NPU (Challenge Target)** | **Priority 1** |
| **Android Devices** | `NNAPIExecutionProvider` | Phone NPU / DSP | **Priority 2** |
| **Apple Silicon (macOS / iOS)** | `CoreMLExecutionProvider` | Apple Neural Engine (ANE) | **Priority 3** |
| **Universal Local Fallback** | `CPUExecutionProvider` | Host x86_64 / ARM CPU | **Priority 4** |

When running on standard local development machines without Snapdragon NPU hardware, the wrapper automatically selects `CPUExecutionProvider` with zero errors.

---

## 4. Benchmark & Validation Results

*All values measured locally on the 20-image synthetic benchmark set (see [`phase1_results.md`](phase1_results.md)).*

### Model Quantization Benchmark

| Metric | FP32 Static ONNX | INT8 Quantized ONNX (`detector_quantized.onnx`) |
|---|:---:|:---:|
| **Input Shape** | `[1, 3, 640, 640]` | `[1, 3, 640, 640]` (Fixed static) |
| **File Size** | 4.54 MB | **1.27 MB** |
| **Compression Ratio** | Baseline | **3.57x (72.0% size reduction)** |
| **Mean Absolute Error (MAE)** | 0.000000 | **0.003383 (< 0.35% drift)** |

### PII Classification Performance

| PII Category | Validation Method | Ground Truth | True Positives | Precision | Recall | F1 Score |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **AADHAAR** | Regex `\d{4}\s?\d{4}\s?\d{4}` + **Verhoeff checksum** | 5 | 5 | **100.0%** | **100.0%** | **1.000** |
| **PAN** | Regex `[A-Z]{5}\d{4}[A-Z]{1}` | 5 | 5 | **100.0%** | **100.0%** | **1.000** |
| **UPI_ID** | Known bank handle whitelist (`@okhdfcbank`, `@oksbi`, etc.) | 8 | 8 | **100.0%** | **100.0%** | **1.000** |
| **PHONE_IN** | Regex `(\+91[\-\s]?)?[6-9]\d{9}` | 12 | 12 | **100.0%** | **100.0%** | **1.000** |
| **IFSC** | Regex `[A-Z]{4}0[A-Z0-9]{6}` | 5 | 5 | **100.0%** | **100.0%** | **1.000** |
| **EMAIL** | RFC 5322 Standard Pattern | 11 | 11 | **100.0%** | **100.0%** | **1.000** |
| **CARD_NUMBER** | Regex + **Luhn checksum** | 5 | 3 | **100.0%** | **60.0%** | **0.750** |
| **OVERALL** | **Full Pipeline** | **51** | **49** | **100.0%** | **96.1%** | **0.980** |

*False Positive Rate on Clean Negative Control Screens:* **0.0% (0 false alarms triggered across telemetry, code editor, API docs, and system settings screens).**

---

## 5. Quickstart Guide

### 1. Installation
```bash
git clone https://github.com/Vijajraj/Screen_PII_Redactor.git
cd Screen_PII_Redactor

pip install -r requirements.txt
```

### 2. Run Automated Test Suite
```bash
python -m pytest tests -v
```

### 3. Generate Synthetic Benchmark Dataset
```bash
python generate_synthetic_data.py
```

### 4. Run Benchmark & Evaluation
```bash
python evaluate_pipeline.py
```

### 5. Launch Jupyter Notebook
```bash
jupyter notebook phase1_notebook.ipynb
```

---

## 6. Code Examples

### Standalone PII Classifier (Zero Dependencies)
```python
from pii_classifier import classify_text

text = "Customer KYC: Aadhaar 9876 5432 1012, PAN ABCDE1234F, UPI priya@okaxis"
results = classify_text(text)

for r in results:
    print(f"Detected {r['pii_type']} ({r['confidence']}): '{r['matched_text']}'")
```

### End-to-End Screen Inference
```python
from inference_wrapper import ScreenPIIPipeline

# Automatically picks QNNExecutionProvider on Snapdragon, or falls back to CPU
pipeline = ScreenPIIPipeline(detector_path="detector_quantized.onnx")
results = pipeline.run_on_image("synthetic_test_set/email_client_01.png")

print(f"Active Execution Provider: {results['execution_provider']}")
print(f"Detected PII entities: {results['pii_findings']}")
```

---

## 7. License

MIT License — Copyright (c) 2026 vijayraj. See [LICENSE](LICENSE) for details.
