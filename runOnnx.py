import cv2
import numpy as np
import onnxruntime as ort
import time

# Config
onnx_model_path = "yolo11n.onnx"
input_size = 320  # Kích thước model ONNX xuất (resize ảnh đầu vào)

# Load model
session = ort.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name

def preprocess(frame, size):
    h, w = frame.shape[:2]
    img = cv2.resize(frame, (size, size))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    return img, w, h

def scale_coords(box, w_orig, h_orig, size):
    # box = [x1, y1, x2, y2] trên ảnh input size x size
    # scale về ảnh gốc
    x_scale = w_orig / size
    y_scale = h_orig / size
    x1, y1, x2, y2 = box
    x1 = int(x1 * x_scale)
    y1 = int(y1 * y_scale)
    x2 = int(x2 * x_scale)
    y2 = int(y2 * y_scale)
    return x1, y1, x2, y2

def main(video_source=0):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Cannot open video source", video_source)
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        st_time = time.time()
        input_tensor, w_orig, h_orig = preprocess(frame, input_size)
        outputs = session.run(None, {input_name: input_tensor})
        detections = outputs[0]  # shape: [1, N, 6]

        for det in detections[0]:
            x1, y1, x2, y2, conf, cls_id = det
            if conf < 0.3:
                continue

            x1, y1, x2, y2 = scale_coords([x1, y1, x2, y2], w_orig, h_orig, input_size)
            cls_id = int(cls_id)

            label = f"{cls_id}:{conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(15, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("YOLOv8 ONNX", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Nhấn ESC để thoát
            break

        end_time = time.time()
        print(f"Tacttime: {(end_time-st_time)*1000} ms")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(0)  # hoặc main("video.mp4") nếu muốn chạy file video
