import cv2
import numpy as np
from openvino.runtime import Core

# Cấu hình
MODEL_PATH = "openvino_model_nanodet/nanodet-plus-m_416_fp16.xml"  # Đường dẫn đến file model OpenVINO
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.6
INPUT_SIZE = (416, 416)  # Kích thước input của model (width, height)


class NanoDetOpenVINO:
    def __init__(self, model_path, conf_threshold=0.5, nms_threshold=0.6):
        # Khởi tạo OpenVINO runtime
        self.core = Core()
        self.model = self.core.read_model(model_path)
        self.compiled_model = self.core.compile_model(self.model, "CPU")

        # Lấy input và output layers
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        # Lấy thông tin input size
        self.input_size = (self.input_layer.shape[3], self.input_layer.shape[2])  # (width, height)

        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        # Lấy số class từ output shape
        output_shape = self.output_layer.shape
        self.num_classes = output_shape[2] - 4
        print(f"Model loaded: Input size {self.input_size}, Output shape {output_shape}, Classes: {self.num_classes}")

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.input_size)

        # Chuẩn hóa
        image = image.astype(np.float32) / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

        # Chuyển đổi dimension thành (1, C, H, W)
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0)
        return image

    def postprocess(self, predictions, frame_shape):
        output = predictions[0]  # Shape: (num_detections, 4 + num_classes)
        boxes = []
        scores = []
        class_ids = []

        h, w = frame_shape

        for detection in output:
            # Lấy class score
            class_scores = detection[4:]
            class_id = np.argmax(class_scores)
            class_score = class_scores[class_id]

            if class_score < self.conf_threshold:
                continue

            # Lấy thông tin bounding box
            cx, cy, dw, dh = detection[:4]

            # Decode tọa độ
            x1 = max(0, (cx - dw / 2) * w)
            y1 = max(0, (cy - dh / 2) * h)
            x2 = min(w, (cx + dw / 2) * w)
            y2 = min(h, (cy + dh / 2) * h)

            width = x2 - x1
            height = y2 - y1

            # Lưu theo định dạng [x, y, width, height] cho NMS
            boxes.append([x1, y1, width, height])
            scores.append(class_score)
            class_ids.append(class_id)

        # Áp dụng NMS
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.nms_threshold)
        else:
            indices = []

        final_boxes = []
        final_scores = []
        final_class_ids = []

        if indices is not None and len(indices) > 0:
            for i in indices.flatten():
                # Chuyển đổi lại định dạng [x1, y1, x2, y2] để vẽ
                x, y, width, height = boxes[i]
                final_boxes.append([x, y, x + width, y + height])
                final_scores.append(scores[i])
                final_class_ids.append(class_ids[i])

        return final_boxes, final_scores, final_class_ids

    def detect(self, image):
        original_shape = image.shape[:2]
        input_tensor = self.preprocess(image)

        # Chạy inference
        predictions = self.compiled_model([input_tensor])[self.output_layer]

        # Xử lý kết quả
        return self.postprocess(predictions, original_shape)


# Hàm chính
def main():
    # Khởi tạo detector
    detector = NanoDetOpenVINO(MODEL_PATH)

    # Khởi tạo webcam
    cap = cv2.VideoCapture(0)

    # Kiểm tra xem webcam có mở được không
    if not cap.isOpened():
        print("Không thể mở webcam")
        return

    print("Nhấn 'q' để thoát")

    while True:
        # Đọc frame từ webcam
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc frame từ webcam")
            break

        # Phát hiện đối tượng
        boxes, scores, class_ids = detector.detect(frame)

        # Vẽ kết quả
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class {class_id}: {score:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Hiển thị FPS
        cv2.putText(frame, "NanoDet + OpenVINO", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Hiển thị frame
        cv2.imshow('NanoDet OpenVINO Detection', frame)

        # Thoát nếu nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()