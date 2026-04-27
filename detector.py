import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.dirname(os.path.abspath(__file__)))

import cv2
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
_tracking_model = None


def get_tracking_model() -> YOLO:
    global _tracking_model

    if _tracking_model is not None:
        return _tracking_model

    if not os.path.exists(DEFAULT_MODEL_PATH):
        raise FileNotFoundError(
            "Missing YOLO weights file 'yolov8n.pt'. Place it in the project root before processing videos."
        )

    _tracking_model = YOLO(DEFAULT_MODEL_PATH)
    return _tracking_model


def track_objects(model: YOLO, frame):
    results = model.track(
        frame,
        persist=True,
        verbose=False,
        imgsz=640,
        tracker="bytetrack.yaml",
    )
    detections = []

    if not results:
        return frame, detections

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        track_ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            detections.append({
                "label": label,
                "confidence": round(confidence, 2),
                "track_id": track_id,
                "box": [int(x1), int(y1), int(x2), int(y2)],
            })

            id_text = f"{label} ID:{track_id}" if track_id is not None else label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{id_text} {confidence:.2f}",
                (int(x1), max(int(y1) - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    return frame, detections
