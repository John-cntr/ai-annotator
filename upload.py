import os
import uuid
import glob
import json
from fastapi import APIRouter, HTTPException, UploadFile, File
from video_processor import process_video

router = APIRouter()
search_router = APIRouter()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "outputs")

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    # 1. Generate unique IDs and paths
    file_id = str(uuid.uuid4())
    # Provide a fallback string in case file.filename is unexpectedly None
    safe_filename = (file.filename or "video.mp4").replace(" ", "_")
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}_{safe_filename}")
    output_video_path = os.path.join(OUTPUT_DIR, f"{file_id}_annotated.mp4")
    output_json_path = os.path.join(OUTPUT_DIR, f"{file_id}_annotations.json")

    # 2. Save the uploaded file
    content = await file.read()
    with open(input_path, "wb") as out_file:
        out_file.write(content)

    # 3. Process the video
    try:
        attendance = process_video(input_path, output_video_path, output_json_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

    # 4. Return the URLs to access the processed files
    return {
        "message": "Video processed successfully!",
        "video_url": f"/outputs/{file_id}_annotated.mp4",
        "json_url": f"/outputs/{file_id}_annotations.json",
        "attendance": attendance
    }

@router.get("/latest-result")
async def get_latest_result():
    """Returns the most recently processed JSON file to populate the Dashboard."""
    json_files = glob.glob(os.path.join(OUTPUT_DIR, "*_annotations.json"))
    if not json_files:
        raise HTTPException(status_code=404, detail="No results found.")
    
    latest_file = max(json_files, key=os.path.getmtime)
    with open(latest_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    file_id = os.path.basename(latest_file).replace("_annotations.json", "")
    data["video_url"] = f"/outputs/{file_id}_annotated.mp4"
    data["json_url"] = f"/outputs/{file_id}_annotations.json"
    return data

@search_router.get("/search")
async def search_detections(label: str):
    """Placeholder for the Phase 2 Search API."""
    return {
        "label": label,
        "total_timestamps": 0,
        "results": []
    }
