"""Extract real labeled glyphs BEFORE training; inherit parent image splits."""
import argparse
import json
from pathlib import Path
import cv2
from config import ALPHABET
from src.preprocessing import read_image,grayscale,normalize_character
from .data_tools import save_json

def prepare(data: Path,out: Path,alphabet: str):
    if (out/"train.json").exists():
        raise ValueError("Output already exists.")
    counts={}
    for split in ("train","val","test"):
        rows=[]
        source=json.loads((data/f"{split}.json").read_text(encoding="utf-8"))
        for i,r in enumerate(source):
            image=grayscale(read_image(data/r["image"]))
            for j,c in enumerate(r.get("characters",[])):
                label=c["label"]
                if label not in alphabet:
                    continue
                x1,y1,x2,y2=map(int,c["box"])
                if not(0<=x1<x2<=image.shape[1] and 0<=y1<y2<=image.shape[0]):
                    raise ValueError("Invalid character box.")
                crop=image[y1:y2,x1:x2]
                binary=cv2.threshold(crop,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
                # Most plate crops have light background; label manifest can override.
                if c.get("polarity","dark")=="dark":
                    binary=255-binary
                glyph=normalize_character(binary)
                path=out/"images"/f"{split}_{i:06d}_{j:02d}.png"
                path.parent.mkdir(parents=True,exist_ok=True)
                cv2.imwrite(str(path),glyph)
                rows.append({"image":f"images/{path.name}","label":label,"group":r["group"]})
        if not rows:
            raise ValueError(f"No character annotations in {split}; add characters to the source manifest.")
        save_json(out/f"{split}.json",rows)
        counts[split]=len(rows)
    card=json.loads((data/"dataset_card.json").read_text(encoding="utf-8"))
    save_json(out/"dataset_card.json",{**card,"counts":counts,"note":"Inherited parent image splits"})
    save_json(out/"alphabet.json",alphabet)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--alphabet",default=ALPHABET)
    a=p.parse_args()
    prepare(a.data,a.out,a.alphabet)

if __name__=="__main__":
    main()
