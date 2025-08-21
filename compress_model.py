from openvino.runtime import Core, serialize
from nncf import Dataset, quantize
import numpy as np
import os

# === Load model ===
core = Core()
model = core.read_model("yolo11n_int8_openvino_model/yolo11n.xml")
input_layer = model.input(0)
input_name = input_layer.get_any_name()

# === Define shape ===
shape = [1, 3, 320, 320]  # hoặc lấy từ input_layer.partial_shape

# === Generate random data ===
num_samples = 50
data = [np.random.rand(*shape).astype(np.float32) for _ in range(num_samples)]

# === Create NNCF Dataset wrapper ===
def preprocess_fn(tensor):
    return {input_name: tensor}

nncf_dataset = Dataset(data, preprocess_fn)

# === Quantize ===
quantized_model = quantize(model, nncf_dataset)

# === Save model ===
os.makedirs("openvino_model_320", exist_ok=True)
serialize(quantized_model,
          "openvino_model_320/yolo11nu_int8.xml",
          "openvino_model_320/yolo11nu_int8.bin")

print("✅ Đã quantize với NNCF Dataset wrapper.")
