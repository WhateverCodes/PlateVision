"""Launch the local training-glyph review page; no training or model downloads."""
from pathlib import Path
import subprocess
import sys
import urllib.request
import webbrowser


def main():
    url='http://127.0.0.1:8506'
    try:
        with urllib.request.urlopen(url+'/_stcore/health',timeout=2) as response:
            if response.read().strip()==b'ok':
                webbrowser.open(url);return 0
    except OSError:pass
    print('Opening character review. Keep this window open while checking characters.')
    try:
        return subprocess.call([sys.executable,'-m','streamlit','run','review_glyphs.py',
            '--server.address','127.0.0.1','--server.port','8506','--server.headless','false'],
            cwd=Path(__file__).resolve().parent)
    except KeyboardInterrupt:return 0


if __name__=='__main__':raise SystemExit(main())
