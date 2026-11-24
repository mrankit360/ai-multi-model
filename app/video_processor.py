import cv2
import os

def extract_video_frames(video_path, interval=2):
    output_dir = "extracted_frames"
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Unable to open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps else 0

    if interval <= 0:
        raise ValueError("Interval must be greater than zero")

    frames = []
    timestamp = 0

    while timestamp <= duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        success, frame = cap.read()

        if not success:
            break

        filename = f"frame_{timestamp:.2f}.jpg"
        filepath = os.path.join(output_dir, filename)

        cv2.imwrite(filepath, frame)

        frames.append({
            "timestamp": round(timestamp, 2),
            "frame_path": filepath
        })

        timestamp += interval

    cap.release()

    return {
        "duration": round(duration, 2),
        "fps": fps,
        "frames": frames
    }