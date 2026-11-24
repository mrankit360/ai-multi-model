def calculate_iou(box1, box2):
    x_left = max(box1["x1"], box2["x1"])
    y_top = max(box1["y1"], box2["y1"])
    x_right = min(box1["x2"], box2["x2"])
    y_bottom = min(box1["y2"], box2["y2"])

    intersection = max(0, x_right - x_left) * \
                   max(0, y_bottom - y_top)

    area1 = (
        (box1["x2"] - box1["x1"]) *
        (box1["y2"] - box1["y1"])
    )

    area2 = (
        (box2["x2"] - box2["x1"]) *
        (box2["y2"] - box2["y1"])
    )

    union = area1 + area2 - intersection

    return intersection / union if union else 0


def track_objects(frame_results, threshold=0.3):
    tracks = []
    next_id = 1

    for frame in frame_results:
        current = frame["detections"]

        for detection in current:
            best_track = None
            best_iou = threshold

            for track in tracks:
                if track["object"] != detection["object"]:
                    continue

                if track["last_timestamp"] >= frame["timestamp"]:
                    continue

                iou = calculate_iou(
                    track["last_box"],
                    detection["bounding_box"]
                )

                if iou > best_iou:
                    best_iou = iou
                    best_track = track

            if best_track:
                best_track["timestamps"].append(
                    frame["timestamp"]
                )
                best_track["last_box"] = detection["bounding_box"]
                best_track["last_timestamp"] = frame["timestamp"]
            else:
                tracks.append({
                    "track_id": next_id,
                    "object": detection["object"],
                    "timestamps": [frame["timestamp"]],
                    "last_box": detection["bounding_box"],
                    "last_timestamp": frame["timestamp"]
                })
                next_id += 1

    return [
        {
            "track_id": t["track_id"],
            "object": t["object"],
            "timestamps": t["timestamps"],
            "frame_count": len(t["timestamps"])
        }
        for t in tracks
    ]