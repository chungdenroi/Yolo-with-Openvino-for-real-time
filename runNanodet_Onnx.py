import cv2
import numpy as np
import onnxruntime as ort
import time
# Cấu hình
MODEL_PATH = "nanodet/nanodet-plus-m_416.onnx"
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.6
INPUT_SIZE = (416, 416)  # (width, height)

# Khởi tạo ONNX Runtime
session = ort.InferenceSession(MODEL_PATH)

def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, INPUT_SIZE)
    image = image.astype(np.float32) / 255.0
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image = image.transpose(2, 0, 1)  # (C, H, W)
    image = np.expand_dims(image, axis=0)  # (1, C, H, W)
    return image

def postprocess(predictions, frame_shape):
    # output shape: (1, 3598, 33)
    output = predictions[0][0]  # (3598, 33)

    boxes = []
    scores = []
    class_ids = []

    h, w = frame_shape

    for detection in output:
        cx, cy, dw, dh = detection[:4]  # bbox center x,y and w,h (normalized)
        class_scores = detection[4:]    # class scores

        class_score = np.max(class_scores)
        class_id = np.argmax(class_scores)

        if class_score < CONFIDENCE_THRESHOLD:
            continue

        # Chuyển bbox từ (cx, cy, w, h) sang (x1, y1, x2, y2)
        x1 = (cx - dw / 2) * w
        y1 = (cy - dh / 2) * h
        x2 = (cx + dw / 2) * w
        y2 = (cy + dh / 2) * h

        # Đảm bảo kiểu float32 và không phải numpy array 0-dim
        box = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]  # (x, y, w, h) format cho NMSBoxes

        boxes.append(box)
        scores.append(float(class_score))
        class_ids.append(int(class_id))

    if len(boxes) == 0:
        return [], [], []

    # Convert list sang numpy array đúng dtype
    boxes_np = np.array(boxes, dtype=np.float32)
    scores_np = np.array(scores, dtype=np.float32)

    # NMSBoxes cần list hoặc numpy array dạng float32
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    final_boxes = []
    final_scores = []
    final_class_ids = []

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w_box, h_box = boxes[i]
            x1 = int(x)
            y1 = int(y)
            x2 = int(x + w_box)
            y2 = int(y + h_box)

            final_boxes.append([x1, y1, x2, y2])
            final_scores.append(scores[i])
            final_class_ids.append(class_ids[i])

    return final_boxes, final_scores, final_class_ids


# Khởi tạo video capture
cap = cv2.VideoCapture(0)  # 0: webcam, thay bằng file video nếu muốn

while True:
    ret, frame = cap.read()
    if not ret:
        break

    original_shape = frame.shape[:2]  # (height, width)

    input_tensor = preprocess(frame)

    # Chạy inference
    st = time.time()
    input_tensor = input_tensor.astype(np.float32)
    results = session.run(None, {session.get_inputs()[0].name: input_tensor})
    print(f"Time predict: {(time.time()-st)*1000}")

    boxes, scores, class_ids = postprocess(results, original_shape)

    # Vẽ bounding box và nhãn
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Class {class_id}: {score:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("NanoDet Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
