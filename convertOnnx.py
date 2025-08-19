from ultralytics import YOLO

# Load YOLOv8 model (.pt file)
model = YOLO("yolo11n.pt")  # hoặc yolov8s.pt, yolov8m.pt tùy bạn

# Export to ONNX with NMS included
model.export(format="onnx", dynamic=True, opset=12, simplify=True, imgsz=320, nms=True)
