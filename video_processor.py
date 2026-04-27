import os

# Fix for Windows: Prevents crashes when OpenCV and PyTorch load conflicting OpenMP libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json  # noqa: E402

import cv2  # noqa: E402
from ultralytics import YOLO  # noqa: E402
import glob  # noqa: E402
import numpy as np  # noqa: E402
from deepface import DeepFace  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
model = None
known_encodings = []
known_names = []

def load_known_faces():
    global known_encodings, known_names
    if known_encodings: return
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    for filepath in glob.glob(os.path.join(KNOWN_FACES_DIR, "*.*")):
        if filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    # Extract FaceNet embeddings (requires OpenCV detector backend)
                    result = DeepFace.represent(img_path=filepath, model_name="Facenet", enforce_detection=True, detector_backend="opencv")
                    if result and len(result) > 0:
                        known_encodings.append(np.array(result[0]["embedding"]))
                        known_names.append(os.path.splitext(os.path.basename(filepath))[0])
                except Exception as e:
                    print(f"Skipping {filepath} (no face detected): {e}")

load_known_faces()


def get_model():
    global model

    if model is not None:
        return model

    # YOLO automatically downloads the weights if they are not found locally!
    model = YOLO("yolov8n.pt")
    return model

def process_video(input_path: str, output_video_path: str, output_json_path: str):
    loaded_model = get_model()
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    # 1. Open the uploaded video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0  # Fallback in case OpenCV can't detect FPS

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 2. Setup video writer
    # Force Windows Media Foundation (MSMF) backend to avoid FFMPEG libopenh264 errors
    # We use avc1 (H264) so the video plays natively in web browsers!
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, cv2.CAP_MSMF, fourcc, fps, (width, height))
    
    if not out.isOpened():
        # Fallback to standard backend with mp4v if MSMF is unavailable
        out = cv2.VideoWriter(output_video_path, cv2.CAP_ANY, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
    if not out.isOpened():
        cap.release()
        raise ValueError(f"Could not create output video file: {output_video_path}")
    
    annotations = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    track_id_to_name = {}
    attendance = set()

    try:
        # 3. Process frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            timestamp = round(frame_count / fps, 2)
            frame_annotations = []

            # 4. Run YOLO detection
            try:
                # Use track instead of predict for ID association
                results = loaded_model.track(frame, persist=True, verbose=False)
                
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        label = loaded_model.names[cls_id]
                        track_id = int(box.id[0]) if box.id is not None else -1

                        display_name = label
                        if label == "person" and track_id != -1:
                            if track_id in track_id_to_name:
                                display_name = track_id_to_name[track_id]
                            else:
                                face_crop = frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
                                display_name = "Unknown"
                                if face_crop.size > 0:
                                    try:
                                        # DeepFace natively uses BGR crops!
                                        result = DeepFace.represent(img_path=face_crop, model_name="Facenet", enforce_detection=True, detector_backend="opencv")
                                        if result and len(result) > 0:
                                            enc = np.array(result[0]["embedding"])
                                            best_match_idx = -1
                                            best_dist = 1.0 # Facenet Cosine threshold is ~0.40
                                            
                                            # Find the closest match manually using Cosine Distance
                                            for i, known_enc in enumerate(known_encodings):
                                                dist = 1 - np.dot(enc, known_enc) / (np.linalg.norm(enc) * np.linalg.norm(known_enc))
                                                if dist < 0.40 and dist < best_dist:
                                                    best_dist = dist
                                                    best_match_idx = i
                                                    
                                            if best_match_idx != -1:
                                                display_name = known_names[best_match_idx]
                                                attendance.add(display_name)
                                    except Exception:
                                        pass # DeepFace representation failed or no clear face was found in the crop
                                track_id_to_name[track_id] = display_name
                                
                        display_label = f"{display_name} (ID:{track_id})" if track_id != -1 else display_name

                        # Save annotation data
                        frame_annotations.append({
                            "label": display_name,
                            "confidence": round(conf, 2),
                            "box": [int(x1), int(y1), int(x2), int(y2)]
                        })

                        # 5. Draw bounding box and label on the frame
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"{display_label} {conf:.2f}",
                            (int(x1), max(int(y1) - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )
            except Exception as exc:
                print(f"Skipping frame {frame_count} due to YOLO/OpenCV error: {exc}")

            annotations.append({
                "timestamp": timestamp,
                "frame": frame_count,
                "detections": frame_annotations
            })
            out.write(frame)
            frame_count += 1
            
            # Print dynamic progress on the same line
            if total_frames > 0:
                print(f"\rProcessing frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)", end="", flush=True)
            else:
                print(f"\rProcessing frame {frame_count}...", end="", flush=True)
    finally:
        print()  # Move to the next line after processing is complete so we don't overwrite it
        cap.release()
        out.release()

    # 6. Save annotations to JSON
    output_payload = {
        "source_video": os.path.basename(input_path),
        "annotated_video": os.path.basename(output_video_path),
        "fps": fps,
        "frames_processed": frame_count,
        "annotations": annotations,
    }
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=4)
        
    attendance_list = list(attendance)
    attendance_path = os.path.join(os.path.dirname(output_json_path), "attendance.json")
    with open(attendance_path, "w", encoding="utf-8") as f:
        json.dump({"present": attendance_list, "total_detected": len(attendance_list)}, f, indent=4)
        
    return attendance_list
