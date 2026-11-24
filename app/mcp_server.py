
import logging
from mcp.server.fastmcp import FastMCP

# IMPORTANT: Logs must go to stderr, not stdout.
logging.basicConfig(level=logging.INFO)

mcp = FastMCP("Video Intelligence Server")


@mcp.tool()
def extract_video_frames(
    video_path: str,
    interval: int = 2
) -> dict:
    """Extract frames from an MP4 video."""

    from app.video_processor import extract_video_frames as extract

    return extract(video_path, interval)


@mcp.tool()
def detect_objects(
    frame_path: str,
    confidence: float = 0.4
) -> dict:
    """Detect objects in an extracted frame."""

    from app.object_detector import detect_objects as detect

    return detect(frame_path, confidence)


if __name__ == "__main__":
    mcp.run(transport="stdio")
