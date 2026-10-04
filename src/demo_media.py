"""Load a local development example without hard-coding recognition output."""
from io import BytesIO
from pathlib import Path
import json

class LocalMedia(BytesIO):
    def __init__(self,path):
        data=Path(path).read_bytes();super().__init__(data)
        self.name=Path(path).name;self.size=len(data)

def example_path(root):
    bundled=Path(root)/"assets/development_example.png"
    if bundled.exists():return bundled
    folder=Path(root)/"data/detector_real_v1"
    manifest=folder/"val.json"
    if not manifest.exists():return None
    for row in json.loads(manifest.read_text()):
        if row.get("source_file")=="dc_license_plates_7K9NEIDD2KK46L6F.jpg":
            path=folder/row["image"]
            return path if path.exists() else None
    return None
