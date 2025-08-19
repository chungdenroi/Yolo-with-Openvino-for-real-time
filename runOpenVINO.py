import cv2
import numpy as np
import time
from openvino.runtime import Core
from  short import  Sort

# ==== CONFIG ====
# model_path = "openvino_model_int8/yolov8n_int8.xml"
model_path = "openvino_model/yolov8n_fp16.xml"
# model_path = "openvino_model_640_int8/yolov8n_int8.xml"
# model_path = "openvino_model_320/yolov5nu_fp16.xml"
# model_path = "openvino_model_320/yolov5nu_int8.xml"
# model_path = "openvino_model_yolov10/yolov10n_fp16.xml"
model_path = "openvino_model_yolov11/yolo11n_fp16.xml"
input_size = 320  # Resize ảnh đầu vào giống khi export

# ==== LOAD OPENVINO MODEL ====
core = Core()
model = core.read_model(model_path)
compiled_model = core.compile_model(model, "GPU")

input_layer = compiled_model.input(0)
input_name = input_layer.get_any_name()

tracker = Sort()

# ==== PREPROCESS ====
def preprocess(frame, size):
    h, w = frame.shape[:2]
    img = cv2.resize(frame, (size, size))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC → CHW
    img = np.expand_dims(img, axis=0)
    return img, w, h

# ==== SCALE COORDINATES ====
def scale_coords(box, w_orig, h_orig, size):
    x_scale = w_orig / size
    y_scale = h_orig / size
    x1, y1, x2, y2 = box
    x1 = int(x1 * x_scale)
    y1 = int(y1 * y_scale)
    x2 = int(x2 * x_scale)
    y2 = int(y2 * y_scale)
    return x1, y1, x2, y2

# ==== MAIN INFERENCE LOOP ====
def main(video_source=0):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("❌ Cannot open video source:", video_source)
        return

    fps = 0
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        st_time = time.time()
        input_tensor, w_orig, h_orig = preprocess(frame, input_size)

        # Inference
        results = compiled_model({input_name: input_tensor})  # Dict[str: np.ndarray]
        detections = list(results.values())[0]  # shape: [1, N, 6]
        detections_list = []
        for det in detections[0]:  # batch dim = 1
            x1, y1, x2, y2, conf, cls_id = det
            if conf < 0.3:
                continue

            x1, y1, x2, y2 = scale_coords([x1, y1, x2, y2], w_orig, h_orig, input_size)
            cls_id = int(cls_id)
            detections_list.append([x1, y1, x2, y2, conf])

            # label = f"{cls_id}:{conf:.2f}"
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(frame, label, (x1, max(15, y1-5)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if detections_list:
            dets = np.array(detections_list)
        else:
            dets = np.zeros((0, 5))

        tracks = tracker.update(dets)
        for x1, y1, x2, y2, track_id in tracks.astype(int):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        frame_count += 1
        elapsed_time = time.time() - start_time
        print(f"Tacttime: {(time.time()-st_time)*1000} ms")

        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        # Vẽ FPS lên frame
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("YOLOv8 - OpenVINO", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC để thoát
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
