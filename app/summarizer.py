from collections import Counter

def generate_summary(frame_results, tracks):
    summary = []

    for frame in frame_results:
        objects = [
            item["object"]
            for item in frame["detections"]
        ]

        if objects:
            counts = Counter(objects)

            description = ", ".join(
                f"{name} ({count})"
                for name, count in counts.items()
            )
        else:
            description = "No objects detected"

        summary.append({
            "timestamp": frame["timestamp"],
            "description": (
                f"At {frame['timestamp']} seconds: "
                f"{description}"
            )
        })

    return {
        "total_frames_analyzed": len(frame_results),
        "unique_object_types": sorted({
            t["object"] for t in tracks
        }),
        "total_tracked_objects": len(tracks),
        "timeline": summary
    }