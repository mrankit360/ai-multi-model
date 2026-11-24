from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import os
import uuid
import shutil
import json

from app.video_processor import extract_video_frames
from app.object_detector import detect_objects
from app.tracker import track_objects
from app.summarizer import generate_summary
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="AI Video Intelligence System",
    description="Video analysis using YOLOv8 and MCP"
)

for folder in [
    "uploads",
    "extracted_frames",
    "annotated_frames",
    "reports"
]:
    os.makedirs(folder, exist_ok=True)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    with open("templates/index.html") as f:
        return f.read()

@app.get("/")
def home():
    return {
        "message": "AI Video Intelligence API is running"
    }


@app.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    interval: int = Form(2),
    confidence: float = Form(0.4)
):

    if not file.filename.lower().endswith(".mp4"):
        raise HTTPException(
            status_code=400,
            detail="Only MP4 videos are supported"
        )

    if interval <= 0:
        raise HTTPException(
            status_code=400,
            detail="Interval must be positive"
        )

    if not 0 < confidence <= 1:
        raise HTTPException(
            status_code=400,
            detail="Confidence must be between 0 and 1"
        )

    video_id = str(uuid.uuid4())

    video_path = os.path.join(
        "uploads",
        f"{video_id}.mp4"
    )

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Step 1: Extract frames
        extraction = extract_video_frames(
            video_path,
            interval
        )

        frame_results = []

        # Step 2: Detect objects
        for frame in extraction["frames"]:

            result = detect_objects(
                frame["frame_path"],
                confidence
            )

            frame_results.append({
                "timestamp": frame["timestamp"],
                "frame_path": frame["frame_path"],
                "annotated_frame": result["annotated_frame"],
                "detections": result["detections"]
            })

        # Step 3: Track objects
        tracks = track_objects(frame_results)

        # Step 4: Generate summary
        summary = generate_summary(
            frame_results,
            tracks
        )

        # Step 5: Generate JSON report
        report = {
            "video_id": video_id,
            "video_name": file.filename,
            "video_duration": extraction["duration"],
            "fps": extraction["fps"],
            "interval": interval,
            "summary": summary,
            "object_tracks": tracks,
            "frames": frame_results
        }

        report_path = os.path.join(
            "reports",
            f"{video_id}.json"
        )

        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)

        return report

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/reports/{video_id}")
def get_report(video_id: str):

    report_path = os.path.join(
        "reports",
        f"{video_id}.json"
    )

    if not os.path.exists(report_path):
        raise HTTPException(
            status_code=404,
            detail="Report not found"
        )

    return FileResponse(
        report_path,
        media_type="application/json",
        filename=f"{video_id}.json"
    )


app.mount(
    "/annotated_frames",
    StaticFiles(directory="annotated_frames"),
    name="annotated_frames"
)