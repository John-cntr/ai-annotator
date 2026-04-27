import json
import os
import re
import shutil
import subprocess
import wave

import cv2
import imageio_ffmpeg
import numpy as np
import whisper

from detector import get_tracking_model, track_objects

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "outputs")
LATEST_RESULT_PATH = os.path.join(OUTPUT_DIR, "result.json")
LATEST_TRANSCRIPT_PATH = os.path.join(OUTPUT_DIR, "transcript.txt")
_whisper_model = None
MAX_FRAME_DIMENSION = 960


def get_whisper_model():
    global _whisper_model

    if _whisper_model is None:
        _whisper_model = whisper.load_model("tiny")

    return _whisper_model


def get_ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def extract_audio(video_path: str, audio_path: str) -> None:
    ffmpeg_exe = get_ffmpeg_exe()
    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        audio_path,
    ]
    subprocess.run(command, check=True, capture_output=True)


def convert_to_browser_video(temp_video_path: str, output_video_path: str, fps: float) -> None:
    command = [
        get_ffmpeg_exe(),
        "-y",
        "-i",
        temp_video_path,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-r",
        str(fps),
        output_video_path,
    ]

    try:
        subprocess.run(command, check=True, capture_output=True)
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
    except Exception:
        if os.path.exists(output_video_path):
            os.remove(output_video_path)
        shutil.move(temp_video_path, output_video_path)


def ensure_browser_playable_video(video_path: str) -> None:
    if not os.path.exists(video_path):
        return

    probe_command = [get_ffmpeg_exe(), "-i", video_path]
    probe = subprocess.run(probe_command, capture_output=True, text=True)
    stream_info = (probe.stderr or "") + (probe.stdout or "")

    if "h264" in stream_info.lower() and "yuv420p" in stream_info.lower():
        return

    temp_fixed_path = video_path.replace(".mp4", "_fixed.mp4")
    convert_to_browser_video(video_path, temp_fixed_path, 30.0)
    if os.path.exists(video_path):
        os.remove(video_path)
    shutil.move(temp_fixed_path, video_path)


def load_wav_audio(audio_path: str) -> np.ndarray:
    with wave.open(audio_path, "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def get_scaled_dimensions(width: int, height: int) -> tuple[int, int]:
    largest_side = max(width, height)
    if largest_side <= MAX_FRAME_DIMENSION:
        return width, height

    scale = MAX_FRAME_DIMENSION / float(largest_side)
    scaled_width = max(2, int(width * scale))
    scaled_height = max(2, int(height * scale))

    if scaled_width % 2 != 0:
        scaled_width -= 1
    if scaled_height % 2 != 0:
        scaled_height -= 1

    return scaled_width, scaled_height


def generate_transcript(video_path: str, transcript_path: str) -> str:
    audio_path = os.path.splitext(transcript_path)[0] + ".wav"
    transcript = ""

    try:
        extract_audio(video_path, audio_path)
        model = get_whisper_model()
        result = model.transcribe(
            audio_path,
            fp16=False,
            language="en",
            condition_on_previous_text=False,
            temperature=0,
            no_speech_threshold=0.6,
        )
        transcript = result.get("text", "").strip()
    except Exception:
        transcript = ""

    if len(transcript) < 3 or (len(transcript) <= 8 and re.fullmatch(r"[A-Z0-9]+", transcript or "")):
        transcript = "No clear speech detected."

    with open(transcript_path, "w", encoding="utf-8") as transcript_file:
        transcript_file.write(transcript)

    if os.path.exists(audio_path):
        os.remove(audio_path)

    return transcript


def process_video(input_path: str, output_video_path: str, output_json_path: str):
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    model = get_tracking_model()
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_width, output_height = get_scaled_dimensions(width, height)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    print(f"[PROCESS] Input: {os.path.basename(input_path)}")
    print(f"[PROCESS] Source resolution: {width}x{height}")
    print(f"[PROCESS] Output resolution: {output_width}x{output_height}")
    print(f"[PROCESS] FPS: {fps:.2f}")
    if total_frames:
        print(f"[PROCESS] Total frames: {total_frames}")

    temp_output_video_path = output_video_path.replace(".mp4", "_temp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_output_video_path, fourcc, fps, (output_width, output_height))
    if not out.isOpened():
        cap.release()
        raise ValueError(f"Could not create output video file: {temp_output_video_path}")

    annotations = []
    frame_count = 0
    progress_step = max(1, total_frames // 20) if total_frames else 30

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if (output_width, output_height) != (width, height):
                frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)

            timestamp = round(frame_count / fps, 2)
            annotated_frame, detections = track_objects(model, frame)
            annotations.append({
                "timestamp": timestamp,
                "frame": frame_count,
                "detections": detections,
            })
            out.write(annotated_frame)
            frame_count += 1

            if total_frames and (frame_count % progress_step == 0 or frame_count == total_frames):
                percent = (frame_count / total_frames) * 100
                print(f"[PROCESS] {percent:6.2f}% ({frame_count}/{total_frames} frames)")
            elif not total_frames and frame_count % progress_step == 0:
                print(f"[PROCESS] Processed {frame_count} frames")
    finally:
        cap.release()
        out.release()

    print("[PROCESS] Converting video for browser playback...")
    convert_to_browser_video(temp_output_video_path, output_video_path, fps)

    transcript_name = os.path.basename(output_json_path).replace("_annotations.json", "_transcript.txt")
    transcript_path = os.path.join(os.path.dirname(output_json_path), transcript_name)
    print("[PROCESS] Generating transcript...")
    transcript = generate_transcript(input_path, transcript_path)
    print("[PROCESS] Transcript complete.")

    payload = {
        "source_video": os.path.basename(input_path),
        "annotated_video": os.path.basename(output_video_path),
        "json_file": os.path.basename(output_json_path),
        "transcript_file": os.path.basename(transcript_path),
        "fps": fps,
        "frames_processed": frame_count,
        "output_resolution": f"{output_width}x{output_height}",
        "transcript": transcript,
        "annotations": annotations,
    }

    with open(output_json_path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=4)

    with open(LATEST_RESULT_PATH, "w", encoding="utf-8") as latest_file:
        json.dump(payload, latest_file, indent=4)

    with open(LATEST_TRANSCRIPT_PATH, "w", encoding="utf-8") as transcript_file:
        transcript_file.write(transcript)

    print(f"[PROCESS] Saved video: {output_video_path}")
    print(f"[PROCESS] Saved JSON: {output_json_path}")
    print(f"[PROCESS] Saved transcript: {transcript_path}")
    return payload


def get_latest_result():
    if not os.path.exists(LATEST_RESULT_PATH):
        raise FileNotFoundError("No processed result found. Upload and process a video first.")

    with open(LATEST_RESULT_PATH, "r", encoding="utf-8") as result_file:
        result = json.load(result_file)

    video_path = os.path.join(OUTPUT_DIR, result.get("annotated_video", ""))
    transcript_path = os.path.join(OUTPUT_DIR, result.get("transcript_file", ""))

    ensure_browser_playable_video(video_path)

    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as transcript_file:
            transcript_text = transcript_file.read().strip()
        if len(transcript_text) < 3 or (len(transcript_text) <= 8 and re.fullmatch(r"[A-Z0-9]+", transcript_text or "")):
            transcript_text = "No clear speech detected."
            with open(transcript_path, "w", encoding="utf-8") as transcript_file:
                transcript_file.write(transcript_text)
            result["transcript"] = transcript_text
            with open(LATEST_RESULT_PATH, "w", encoding="utf-8") as latest_file:
                json.dump(result, latest_file, indent=4)

    return result


def search_by_label(label: str):
    result_data = get_latest_result()

    matches = []
    target_label = label.strip().lower()

    for frame_data in result_data.get("annotations", []):
        detections = [
            {
                "label": detection["label"],
                "track_id": detection.get("track_id"),
                "confidence": detection["confidence"],
            }
            for detection in frame_data.get("detections", [])
            if detection.get("label", "").lower() == target_label
        ]
        if detections:
            matches.append({
                "timestamp": frame_data.get("timestamp"),
                "frame": frame_data.get("frame"),
                "matches": detections,
            })

    return {
        "label": label,
        "total_timestamps": len(matches),
        "results": matches,
    }
