from openvino.tools.mo import convert_model
from openvino.runtime import serialize
import os

# Convert model từ ONNX sang OpenVINO IR (FP16)
model = convert_model(input_model="yolo11n.onnx", compress_to_fp16=True)

# Serialize: lưu model ra file XML + BIN
os.makedirs("openvino_model_yolov11", exist_ok=True)
serialize(model, "openvino_model_yolov11/yolo11n_fp16.xml", "openvino_model_yolov11/yolo11n_fp16.bin")

print("✅ Đã convert và lưu model OpenVINO FP16.")
