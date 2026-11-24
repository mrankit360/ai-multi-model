import cv2
import os
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect_objects(frame_path, confidence=0.4):
    image = cv2.imread(frame_path)

    if image is None:
        raise ValueError("Unable to read frame")

    results = model(image, conf=confidence, verbose=False)

    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            score = float(box.conf[0])

            detections.append({
                "object": class_name,
                "confidence": round(score, 3),
                "bounding_box": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }
            })

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                f"{class_name} {score:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    os.makedirs("annotated_frames", exist_ok=True)

    output_path = os.path.join(
        "annotated_frames",
        os.path.basename(frame_path)
    )

    cv2.imwrite(output_path, image)

    return {
        "detections": detections,
        "annotated_frame": output_path
    }