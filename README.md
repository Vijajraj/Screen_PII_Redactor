# Screen PII Redactor

**On-Device Sensitive PII Detection & Redaction for Indian & Universal Identifiers**  
*Snapdragon AI Lab Challenge — Phase 1: Model Build & Export (Local)*

[![Tests](https://img.shields.io/badge/tests-210%20passed-brightgreen.svg)]()
[![Model](https://img.shields.io/badge/ONNX-Static%20640x640-blue.svg)]()
[![Quantization](https://img.shields.io/badge/INT8-1.27%20MB%20(3.57x)-orange.svg)]()
[![Phase 2](https://img.shields.io/badge/Phase%202-Live%20Capture%20%26%20Demo-success.svg)]()
[![Target EP](https://img.shields.io/badge/Qualcomm-QNN%20NPU-purple.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Overview

**Screen PII Redactor** is a high-performance, on-device privacy engine engineered to detect and redact sensitive Personally Identifiable Information (PII) from screen captures and desktop application frames in real time. Designed specifically for **Qualcomm Snapdragon X-Elite / Hexagon NPU** hardware via the **QNN Execution Provider**, the architecture enforces static tensor dimensions, INT8 dynamic quantization, and deterministic checksum-backed classification.

---

## 2. End-to-End System Architecture

```mermaid
flowchart TD
    subgraph S1["Stage 1: Input & Frame Preprocessing"]
        A["Input Screenshot Frame<br/>(RGB / BGR Array)"] --> B["Bilinear Resizing & Letterboxing<br/>Static Dimensions: 640 x 640"]
        B --> C["Standard Normalization<br/>(img / 255.0 - Mean) / Std<br/>Tensor Shape: [1, 3, 640, 640]"]
    end

    subgraph S2["Stage 2: On-Device Neural Detection"]
        C --> D["PP-OCRv4 Mobile DBNet Text Detector<br/>(detector_quantized.onnx)"]
        D -.->|"Primary Target"| EP1["QNNExecutionProvider<br/>Qualcomm Hexagon NPU"]
        D -.->|"Fallback"| EP2["CPUExecutionProvider<br/>Host x86_64 / ARM CPU"]
        D --> E["Probability Heatmap Output<br/>Tensor Shape: [1, 1, 640, 640]"]
        E --> F["DBNet Postprocessing<br/>Binary Masking (Threshold: 0.3)<br/>Contour Extraction & Polygon Unclipping"]
        F --> G["Rescaled Bounding Boxes<br/>[x1, y1, x2, y2]"]
    end

    subgraph S3["Stage 3: Text Recognition"]
        G --> H["Dynamic Text Region Cropping<br/>Normalized Height: 48px"]
        H --> I["PP-OCRv4 Mobile SVTR Recognizer"]
        I --> J["CTC Greedy Decoding<br/>(Character Dictionary Map)"]
        J --> K["Extracted Raw Text Strings"]
    end

    subgraph S4["Stage 4: Deterministic PII Classification"]
        K --> L["Portable PII Classifier<br/>(pii_classifier.py)"]
        L --> M1["Aadhaar Matcher<br/>Regex + Verhoeff Checksum<br/>Confidence: 1.0"]
        L --> M2["PAN Matcher<br/>Regex Format<br/>Confidence: 0.8"]
        L --> M3["UPI ID Matcher<br/>Regex + Bank Handle Whitelist<br/>Confidence: 0.8"]
        L --> M4["Indian Phone Matcher<br/>Regex Format (+91 / 10-digit)<br/>Confidence: 0.8"]
        L --> M5["IFSC Code Matcher<br/>Regex Format<br/>Confidence: 0.8"]
        L --> M6["Email Matcher<br/>RFC 5322 Standard<br/>Confidence: 0.8"]
        L --> M7["Card Number Matcher<br/>Regex + Luhn Checksum<br/>Confidence: 1.0"]
    end

    subgraph S5["Stage 5: Structured Output"]
        M1 & M2 & M3 & M4 & M5 & M6 & M7 --> N["Sanitized Detection Payload<br/>List of {bbox, pii_type, matched_text, confidence}"]
    end
```

---

## 3. Execution Provider Fallback Sequence

```mermaid
sequenceDiagram
    autonumber
    actor Client as Application Client / Capture Loop
    participant Wrapper as inference_wrapper.py
    participant ORT as ONNX Runtime Environment
    participant Hexagon as Qualcomm Hexagon NPU (QNN)
    participant HostCPU as Host CPU Fallback

    Client->>Wrapper: Initialize Pipeline(detector_quantized.onnx)
    Wrapper->>ORT: Query get_available_providers()
    ORT-->>Wrapper: Return Available Providers List

    alt QNNExecutionProvider is Present (Snapdragon Hardware)
        Wrapper->>Hexagon: Bind Session to QNNExecutionProvider
        Hexagon-->>Wrapper: Hardware Accelerator Ready
    else QNN Not Found (Development / CI Testbed)
        Wrapper->>HostCPU: Fallback to CPUExecutionProvider
        HostCPU-->>Wrapper: CPU Fallback Initialized
    end

    Client->>Wrapper: run_on_image(screenshot.png)
    alt On Snapdragon Device
        Wrapper->>Hexagon: Execute Static Inference [1, 3, 640, 640]
        Hexagon-->>Wrapper: Return Heatmap Tensor [1, 1, 640, 640]
    else On Local Host
        Wrapper->>HostCPU: Execute Static Inference [1, 3, 640, 640]
        HostCPU-->>Wrapper: Return Heatmap Tensor [1, 1, 640, 640]
    end
    Wrapper->>Client: Return Detected Regions & PII Annotations
```

---

## 4. Phase 1 Deliverables (Handoff to Phase 2)

| Number | Deliverable File | Technical Scope | Status |
|:---:|---|---|:---:|
| **1** | [`detector_quantized.onnx`](detector_quantized.onnx) | Final INT8 quantized DBNet text detector with fixed static shape `[1, 3, 640, 640]` (1.27 MB, 3.57x compression) | Completed |
| **2** | [`pii_classifier.py`](pii_classifier.py) | Standalone, portable PII classifier using Python standard library + `re` only (zero external dependencies) | Completed |
| **3** | [`inference_wrapper.py`](inference_wrapper.py) | Hardware-agnostic EP-abstraction inference engine with QNN/NNAPI/CoreML/CPU priority fallback | Completed |
| **4** | [`synthetic_test_set/`](synthetic_test_set/) | 20 synthetic mock screenshots across 5 UI categories + `ground_truth.json` catalog | Completed |
| **5** | [`phase1_results.md`](phase1_results.md) | Measured validation report with real benchmarks (model compression, MAE drift, precision/recall) | Completed |
| **6** | [`phase1_notebook.ipynb`](phase1_notebook.ipynb) | Complete 10-section Jupyter notebook implementing Sections 1-10 of the specification | Completed |

---

## 5. Execution Provider Portability Matrix

The model file and preprocessing pipeline remain invariant across target platforms; only the active execution provider changes:

| Platform | Execution Provider | Target Hardware | Execution Priority |
|---|---|---|:---:|
| **Snapdragon Laptops (Windows)** | `QNNExecutionProvider` | **Qualcomm Hexagon NPU (Challenge Target)** | **Priority 1** |
| **Android Devices** | `NNAPIExecutionProvider` | Mobile NPU / DSP | **Priority 2** |
| **Apple Silicon (macOS / iOS)** | `CoreMLExecutionProvider` | Apple Neural Engine (ANE) | **Priority 3** |
| **Universal Local Fallback** | `CPUExecutionProvider` | Host x86_64 / ARM CPU | **Priority 4** |

---

## 6. Model Quantization & Compression Analysis

```
========================================================================================
MODEL SIZE REDUCTION BREAKDOWN
========================================================================================
FP32 Static ONNX       [========================================] 4.54 MB (Baseline)
INT8 Quantized ONNX    [===========                             ] 1.27 MB (3.57x Compression)
                                                                  72.0% Storage Reduction
========================================================================================
QUANTIZATION ACCURACY DRIFT
========================================================================================
Mean Absolute Error (MAE): 0.003383 (< 0.35% drift across all 20 benchmark test images)
Max Contouring Difference: Localized strictly to sub-pixel edge transitions
========================================================================================
```

| Metric | Pre-Export / FP32 Static ONNX | INT8 Quantized ONNX (`detector_quantized.onnx`) |
|---|:---:|:---:|
| **Tensor Input Shape** | `[1, 3, 640, 640]` (Fixed) | `[1, 3, 640, 640]` (Fixed Static) |
| **Model Size on Disk** | 4.54 MB | **1.27 MB** |
| **Compression Ratio** | Baseline (1.0x) | **3.57x (72.0% size reduction)** |
| **Weight Representation** | Float32 | Dynamic QUInt8 (Quantized Unsigned Int8) |
| **Mean Absolute Error (MAE)** | 0.000000 | **0.003383** |
| **Inference Provider (Local)** | CPUExecutionProvider | CPUExecutionProvider (Fallback Verified) |

---

## 7. Synthetic Dataset Distribution & Classification Benchmarks

### Dataset Category Composition (20 Images, 51 PII Entities)

```mermaid
pie title Synthetic Dataset Category Distribution
    "Email Clients (4 images)" : 12
    "KYC Onboarding Forms (4 images)" : 12
    "Banking Dashboards (4 images)" : 16
    "Customer Support Chat (4 images)" : 11
    "Clean Negative Controls (4 images)" : 0
```

### PII Classification Performance Table

| PII Category | Validation Method & Rules | Ground Truth | True Positives (TP) | False Positives (FP) | Precision | Recall | F1 Score |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **AADHAAR** | Regex `\d{4}\s?\d{4}\s?\d{4}` + **Verhoeff Checksum** | 5 | 5 | 0 | **100.0%** | **100.0%** | **1.000** |
| **PAN** | Regex `[A-Z]{5}\d{4}[A-Z]{1}` | 5 | 5 | 0 | **100.0%** | **100.0%** | **1.000** |
| **UPI_ID** | Known bank handle whitelist (`@okhdfcbank`, `@oksbi`, etc.) | 8 | 8 | 0 | **100.0%** | **100.0%** | **1.000** |
| **PHONE_IN** | Regex `(\+91[\-\s]?)?[6-9]\d{9}` | 12 | 12 | 0 | **100.0%** | **100.0%** | **1.000** |
| **IFSC** | Regex `[A-Z]{4}0[A-Z0-9]{6}` | 5 | 5 | 0 | **100.0%** | **100.0%** | **1.000** |
| **EMAIL** | RFC 5322 Standard Email Pattern | 11 | 11 | 0 | **100.0%** | **100.0%** | **1.000** |
| **CARD_NUMBER** | Regex + **Luhn Checksum** | 5 | 3 | 0 | **100.0%** | **60.0%** | **0.750** |
| **OVERALL** | **Full Pipeline** | **51** | **49** | **0** | **100.0%** | **96.1%** | **0.980** |

### Negative Control Evaluation (False Positive Rate)

| Clean Control Category | Sample Test Image | Evaluated Features | Ground Truth PII | Detected PII (FP) | Error Rate |
|---|---|---|:---:|:---:|:---:|
| **Cluster Telemetry** | `clean_analytics_01.png` | CPU / Memory metrics, P99 latencies | 0 | 0 | **0.0%** |
| **Source Code Editor** | `clean_code_editor_02.png` | Python Dijkstra algorithm implementation | 0 | 0 | **0.0%** |
| **API Documentation** | `clean_docs_page_03.png` | REST endpoints, vector index parameters | 0 | 0 | **0.0%** |
| **Hardware Settings** | `clean_settings_04.png` | Resolution, display frequency, audio bus | 0 | 0 | **0.0%** |

- **Total Clean Screens Evaluated:** 4
- **False Positives Triggered:** **0 (0.0% false-positive rate)**

---

## 8. Deterministic Validation Decision Logic

```mermaid
flowchart LR
    Text["Candidate OCR Text Token"] --> Step1{"Is 12-Digit Numeric Sequence?"}
    Step1 -- Yes --> Step1A{"Passes Verhoeff Algorithm?"}
    Step1A -- Yes --> Res1["Classified: AADHAAR<br/>Confidence: 1.0"]
    Step1A -- No --> Step2

    Step1 -- No --> Step2{"Matches [A-Z]{5}[0-9]{4}[A-Z]?"}
    Step2 -- Yes --> Res2["Classified: PAN<br/>Confidence: 0.8"]
    Step2 -- No --> Step3

    Step3{"Contains @ Handle?"}
    Step3 -- Yes --> Step3A{"Suffix in Known Bank Whitelist?"}
    Step3A -- Yes --> Res3["Classified: UPI_ID<br/>Confidence: 0.8"]
    Step3A -- No --> Step3B{"Matches RFC 5322 Email?"}
    Step3B -- Yes --> Res4["Classified: EMAIL<br/>Confidence: 0.8"]
    Step3B -- No --> Step4

    Step3 -- No --> Step4{"Matches Indian Mobile (+91/6-9)?"}
    Step4 -- Yes --> Res5["Classified: PHONE_IN<br/>Confidence: 0.8"]
    Step4 -- No --> Step5

    Step5{"Matches [A-Z]{4}0[A-Z0-9]{6}?"}
    Step5 -- Yes --> Res6["Classified: IFSC<br/>Confidence: 0.8"]
    Step5 -- No --> Step6

    Step6{"Is 13-19 Digits & Passes Luhn?"}
    Step6 -- Yes --> Res7["Classified: CARD_NUMBER<br/>Confidence: 1.0"]
    Step6 -- No --> Res8["Unclassified / Non-Sensitive Text"]
```

---

## 9. Visual Redaction Demonstration

```
BEFORE REDACTION (Raw Screen Capture):
+-------------------------------------------------------------------------------+
| Identity Verification Form                                                    |
| Name: Rajesh Ramanathan                                                       |
| PAN Number:      ABCDE1234F                                                   |
| Aadhaar UID:     9876 5432 1012                                               |
| Contact Phone:   +91 9123456789                                               |
| Primary UPI:     rajesh@okhdfcbank                                            |
+-------------------------------------------------------------------------------+

AFTER REDACTION (Masked Output):
+-------------------------------------------------------------------------------+
| Identity Verification Form                                                    |
| Name: Rajesh Ramanathan                                                       |
| PAN Number:      [██████████] <- Classified: PAN (conf: 0.8)                  |
| Aadhaar UID:     [██████████████] <- Classified: AADHAAR (conf: 1.0)          |
| Contact Phone:   [██████████████] <- Classified: PHONE_IN (conf: 0.8)         |
| Primary UPI:     [████████████████] <- Classified: UPI_ID (conf: 0.8)         |
+-------------------------------------------------------------------------------+
```

---

## 10. Phase 2: Live Screen Capture & Redaction Application

Phase 2 builds upon the Phase 1 quantized ONNX model and PII classifier to deliver a complete, real-time desktop application.

### Pipeline Overview
```mermaid
flowchart LR
    A["Screen Capture (mss)<br/>Full Screen / Sub-Region"] --> B["Letterbox Preprocessor<br/>Static 640x640 Canvas"]
    B --> C["INT8 DBNet Detector<br/>(CPU / QNN Hexagon)"]
    C --> D["SVTR CTC Recognizer<br/>Text String Extraction"]
    D --> E["Deterministic PII Classifier<br/>Aadhaar, PAN, Card, UPI..."]
    E --> F["Redaction Renderer<br/>Gaussian Blur + Color Badges"]
    F --> G["Live Display & Telemetry<br/>OpenCV HUD Monitor"]
```

### Key Capabilities
- **Fast Desktop Capture (`mss`)**: Low-overhead frame acquisition (< 20 ms) supporting whole desktop or user-defined rectangular sub-regions.
- **Letterbox Transformation**: Preserves text aspect ratios across widescreen and multi-monitor setups without stretching artifacts.
- **Visual Redaction Badges**: High-contrast, color-coded security banners with confidence scoring:
  - `[AADHAAR REDACTED 1.0]` (Crimson)
  - `[PAN REDACTED 0.8]` (Emerald)
  - `[CARD_NUMBER REDACTED 1.0]` (Navy)
  - `[UPI_ID REDACTED 0.8]` (Purple)
  - `[PHONE_IN REDACTED 0.8]` (Orange)
  - `[IFSC REDACTED 0.8]` (Teal)
- **Gaussian Blurring**: Irreversible Gaussian kernel blurring ($k=31$) applied directly over raw sensitive text regions.
- **Autonomous Demo Recording**: Built-in recorder generates both `demo_video.mp4` and `demo_video.gif` showcasing startup, KYC, banking, chat, and false-positive controls.

### Measured Latency Breakdown (CPUExecutionProvider)

| Stage | Mean Latency (ms) | P95 Latency (ms) | Status |
|---|---|---|---|
| Screen Capture (`mss`) | 18.2 ms | 24.1 ms | Optimal |
| Letterbox Preprocessing | 3.8 ms | 5.2 ms | Optimal |
| Model Inference (DBNet + SVTR) | 185.0 ms | 210.4 ms | Within 500ms budget |
| Overlay & Blur Rendering | 2.1 ms | 3.4 ms | Optimal |
| **Total Loop Refresh** | **209.1 ms** | **243.1 ms** | **PASSED (< 500 ms)** |

---

## 11. Quickstart Guide

### 1. Installation
```bash
git clone https://github.com/Vijajraj/Screen_PII_Redactor.git
cd Screen_PII_Redactor

pip install -r requirements.txt
```

### 2. Launch Live Screen PII Redactor
```bash
# Launch interactive live screen monitor (press 'q' to quit, 's' to save snapshot)
python live_capture_app.py --interval 0.5

# Capture a specific window / region (left top width height)
python live_capture_app.py --region 100 100 800 600 --interval 0.5
```

### 3. Generate 30-Second Demo Video & Animated GIF
```bash
python record_demo.py
# Outputs: demo_video.mp4 (2.1 MB) and demo_video.gif (0.23 MB)
```

### 4. Run Phase 2 Live Benchmarks
```bash
python evaluate_phase2.py
# Generates phase2_results.md with real measured latencies and detection stats
```

### 5. Run Full 187-Test Automated Verification Suite
```bash
python -m pytest tests -v
```

### 6. Phase 1 Pipeline Evaluation & Notebook
```bash
python evaluate_pipeline.py
jupyter notebook phase1_notebook.ipynb
```

---

## 12. Code Examples

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

# Automatically selects QNNExecutionProvider on Snapdragon, or falls back to CPU
pipeline = ScreenPIIPipeline(detector_path="detector_quantized.onnx")
results = pipeline.run_on_image("synthetic_test_set/email_client_01.png")

print(f"Active Execution Provider: {results['execution_provider']}")
print(f"Detected PII entities: {results['pii_findings']}")
```

---

## 13. License

MIT License — Copyright (c) 2026 vijayraj. See [LICENSE](LICENSE) for details.
