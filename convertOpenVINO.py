# from openvino.tools.mo import convert_model
# from openvino.runtime import serialize
# import os
#
# # Convert model từ ONNX sang OpenVINO IR (FP16)
# model = convert_model(input_model="yolo11n.onnx", compress_to_fp16=True)
#
# # Serialize: lưu model ra file XML + BIN
# os.makedirs("openvino_model_yolov11", exist_ok=True)
# serialize(model, "openvino_model_yolov11/yolo11n_fp16.xml", "openvino_model_yolov11/yolo11n_fp16.bin")
#
# print("✅ Đã convert và lưu model OpenVINO FP16.")


from ultralytics import YOLO

# Load model
model = YOLO("yolo11n.pt")

# Export OpenVINO FP32 (mặc định)
# model.export(format="openvino", imgsz=640)

# Export OpenVINO FP16
# model.export(format="openvino", imgsz=640, precision="fp16")

# Export OpenVINO INT8
model.export(
    format='openvino',  # hoặc onnx, tensorrt nếu phù hợp phần cứng
    imgsz=320,
    half=True,          # sử dụng FP16 nếu thiết bị hỗ trợ (GPU)
    int8=True,          # sử dụng quantization int8 (nhanh nhất, nhẹ nhất, độ chính xác giảm nhẹ)
    simplify=True,       # giúp đơn giản hóa model (nếu hỗ trợ)
    nms=True
)

