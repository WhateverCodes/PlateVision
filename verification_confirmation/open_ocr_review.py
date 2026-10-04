"""Launch the local plate text review app without starting recognition or training."""
from pathlib import Path
import subprocess,sys,urllib.request,webbrowser

def main():
    url="http://127.0.0.1:8505"
    try:
        with urllib.request.urlopen(url+"/_stcore/health",timeout=2) as r:
            if r.read().strip()==b"ok":
                webbrowser.open(url)
                return 0
    except OSError:
        pass
    print("Opening plate text review. Keep this window open while reviewing.")
    try:
        return subprocess.call([sys.executable,"-m","streamlit","run","review_ocr_labels.py",
                                "--server.address","127.0.0.1","--server.port","8505","--server.headless","false"],
                                cwd=Path(__file__).resolve().parent)
    except KeyboardInterrupt:
        return 0

if __name__=="__main__":
    raise SystemExit(main())
