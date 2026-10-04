"""Import a licensed bounding-box manifest; reject missing provenance/group IDs."""
import argparse
import json
import shutil
from pathlib import Path
from src.preprocessing import read_image
from .data_tools import write_splits,image_hash

def prepare(manifest: Path,out: Path,seed: int):
    raw=json.loads(manifest.read_text(encoding="utf-8"))
    if not raw.get("license") or not raw.get("source"):
        raise ValueError("Manifest must record source and license/permission.")
    if (out/"train.json").exists():
        raise ValueError("Output exists; use a new folder.")
    records=[]
    for i,item in enumerate(raw["records"]):
        if not item.get("group"):
            raise ValueError("Each image requires a capture/vehicle group to prevent leakage.")
        path=(manifest.parent/item["image"]).resolve()
        image=read_image(path)
        h,w=image.shape[:2]
        boxes=item["boxes"]
        for b in boxes:
            if len(b)!=4 or not (0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h):
                raise ValueError(f"Invalid box in {path.name}: {b}")
        texts=item.get("texts",[])
        if texts and len(texts)!=len(boxes):
            raise ValueError("texts must align one-to-one with boxes.")
        dest=out/"images"/f"{i:06d}{path.suffix.lower()}"
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)
        records.append({**item,"image":f"images/{dest.name}","sha256":image_hash(dest)})
    write_splits(out,records,seed,f"{raw['source']} | {raw['license']}")
    print(f"Prepared {len(records)} images in {out}")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    prepare(a.manifest,a.out,a.seed)

if __name__=="__main__":
    main()
