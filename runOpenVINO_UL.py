import cv2
import time
from ultralytics import YOLO

# --- Cấu hình ---
MODEL_PATH = "yolov5nu_int8_openvino_model/"  # Thư mục chứa model OpenVINO (.xml + .bin)
DEVICE = "GPU"  # Bạn có thể thử "intel:gpu" nếu có

IMG_SIZE = 320  # kích thước ảnh input
CONFIDENCE_THRESHOLD = 0.25

# --- Khởi tạo model ---
model = YOLO(MODEL_PATH, task='detect')

# --- Webcam setup ---
cap = cv2.VideoCapture(0)

# Xử lý và đo FPS
fps = 0.0
start_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    t0 = time.time()

    results = model(frame, imgsz=IMG_SIZE, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
    annotated = results.plot()  # vẽ box, label, score

    # Tính FPS trung bình mỗi giây
    if time.time() - start_time >= 1.0:
        fps = frame_count / (time.time() - start_time)
        start_time = time.time()
        frame_count = 0

    # Hiển thị FPS trên ảnh
    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    print(f"Time predict: {(time.time() - t0) *1000} ms")

    cv2.imshow("YOLOv8 OpenVINO Webcam", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
