"""Open PLATEVISION in a browser, reusing an already running local server."""
from pathlib import Path
import subprocess
import sys
import urllib.request
import webbrowser

URL = "http://127.0.0.1:8501"

def main() -> int:
    root = Path(__file__).resolve().parent
    try:
        with urllib.request.urlopen(URL + "/_stcore/health", timeout=2) as response:
            ready = response.read().strip() == b"ok"
    except (OSError, urllib.error.URLError):
        ready = False
    if ready:
        webbrowser.open(URL)
        print("Opened the running app in your browser.")
        return 0
    try:
        import streamlit
    except ImportError:
        print("Setup is incomplete. Please follow the installation steps in README.md.")
        return 1
    print("Opening PLATEVISION. Keep this window open while using the app.")
    print("Press Ctrl+C in this window to stop the app.")
    try:
        return subprocess.call([
            sys.executable, "-m", "streamlit", "run", str(root / "app.py"),
            "--server.port", "8501", "--server.headless", "false",
        ], cwd=root)
    except KeyboardInterrupt:
        return 0

if __name__ == "__main__":
    raise SystemExit(main())
