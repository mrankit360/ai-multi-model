
import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Use the SAME Python interpreter running this client
PYTHON = sys.executable


async def main():

    server_params = StdioServerParameters(
        command=PYTHON,
        args=[
            "-m",
            "app.mcp_server"
        ],
        cwd=str(PROJECT_ROOT)
    )

    print("Starting MCP server...")
    print("Python:", PYTHON)
    print("Project:", PROJECT_ROOT)

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            # Initialize MCP connection
            await session.initialize()

            print("\nMCP connection successful!")

            # List available tools
            tools = await session.list_tools()

            print("\nAvailable MCP Tools:")

            for tool in tools.tools:
                print("-", tool.name)

            # Execute frame extraction
            video_path = str(
                PROJECT_ROOT / "uploads" / "sample.mp4"
            )

            print("\nExecuting extract_video_frames...")

            result = await session.call_tool(
                "extract_video_frames",
                {
                    "video_path": video_path,
                    "interval": 2
                }
            )

            print("\nFrame Extraction Result:")

            for content in result.content:
                if hasattr(content, "text"):
                    print(content.text)

            if result.isError:
                print("Frame extraction tool returned an error.")


if __name__ == "__main__":
    asyncio.run(main())
