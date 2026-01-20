"""
Launch script for the Streamlit web application.

Usage:
    python run_app.py

Or directly:
    streamlit run app/streamlit_app.py
"""
import subprocess
import sys
from pathlib import Path


def main():
    app_path = Path(__file__).parent / "app" / "streamlit_app.py"

    # Launch Streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ])


if __name__ == "__main__":
    main()
