"""Explicit setup download for the separate general-purpose vehicle model."""
import argparse
from pathlib import Path
import torch
from config import DEFAULTS
from src.runtime import configure

URL="https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth"

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=DEFAULTS.vehicle_path)
    a=p.parse_args()
    if a.out.exists():
        print(f"Already present: {a.out}")
        return
    configure()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    tmp=a.out.with_suffix(".download")
    torch.hub.download_url_to_file(URL,str(tmp),hash_prefix="a79551df",progress=True)
    tmp.replace(a.out)
    print("Downloaded general COCO vehicle weights. These do not detect or recognize plates.")

if __name__=="__main__":
    main()
