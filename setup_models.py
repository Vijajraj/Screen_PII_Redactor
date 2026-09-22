"""
setup_models.py — Downloads and prepares ONNX detection & recognition models
Screen PII Redactor (Snapdragon AI Lab Challenge)
"""

import os
import onnx
from onnx import shape_inference
import onnxruntime.quantization as oq
from huggingface_hub import hf_hub_download

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def download_and_convert():
    print("[1/4] Downloading PP-OCRv4 detection ONNX model...")
    det_path = hf_hub_download(
        repo_id="Heliosoph/paddleocr-v4-det-onnx",
        filename="ch_PP-OCRv4_det.onnx",
        local_dir=MODELS_DIR
    )
    
    print("[2/4] Downloading PP-OCRv4 recognition ONNX model and dict...")
    rec_path = hf_hub_download(
        repo_id="xberg-io/paddleocr-onnx-models",
        filename="rec/english/model.onnx",
        local_dir=MODELS_DIR
    )
    dict_path = hf_hub_download(
        repo_id="xberg-io/paddleocr-onnx-models",
        filename="rec/english/dict.txt",
        local_dir=MODELS_DIR
    )

    print("[3/4] Converting detection model to fixed static shape [1, 3, 640, 640]...")
    m = onnx.load(det_path)
    d = m.graph.input[0].type.tensor_type.shape.dim
    d[0].dim_value = 1; d[0].ClearField('dim_param')
    d[1].dim_value = 3; d[1].ClearField('dim_param')
    d[2].dim_value = 640; d[2].ClearField('dim_param')
    d[3].dim_value = 640; d[3].ClearField('dim_param')
    
    # Convert constant weights to initializers for ONNX runtime quantization
    nodes_to_remove = []
    for node in m.graph.node:
        if node.op_type == 'Constant':
            for attr in node.attribute:
                if attr.name == 'value':
                    tensor = attr.t
                    tensor.name = node.output[0]
                    m.graph.initializer.append(tensor)
                    nodes_to_remove.append(node)
                    break
    for n in nodes_to_remove:
        m.graph.node.remove(n)

    static_path = os.path.join(MODELS_DIR, "detector_clean_static.onnx")
    onnx.save(m, static_path)
    print(f"  Saved static model to: {static_path}")

    print("[4/4] Quantizing static model to INT8 (detector_quantized.onnx)...")
    quantized_path = "detector_quantized.onnx"
    oq.quantize_dynamic(
        static_path,
        quantized_path,
        weight_type=oq.QuantType.QUInt8
    )
    
    size_fp32 = os.path.getsize(static_path) / (1024 * 1024)
    size_int8 = os.path.getsize(quantized_path) / (1024 * 1024)
    print(f"  FP32 Size: {size_fp32:.2f} MB")
    print(f"  INT8 Size: {size_int8:.2f} MB (Compression: {size_fp32/size_int8:.2f}x)")
    print(f"  Successfully created {quantized_path}")


if __name__ == "__main__":
    download_and_convert()
