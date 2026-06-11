"""
Launch script for the FastAPI server.

Usage:
    python run_api.py

Environment:
    API_HOST (default 0.0.0.0)
    API_PORT (default 8000)
"""
import os

import uvicorn


def main():
    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
