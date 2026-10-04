"""Small reproducible self-generated datasets. Never represented as real Indian photos."""
import argparse
import random
from pathlib import Path
import cv2
import numpy as np
from config import ROOT,ALPHABET
from src.preprocessing import normalize_character
from .data_tools import save_json,write_splits,image_hash

FONTS=[cv2.FONT_HERSHEY_SIMPLEX,cv2.FONT_HERSHEY_DUPLEX,cv2.FONT_HERSHEY_TRIPLEX]
STATES=["MH","KA","DL","TN","GJ","RJ","UP","WB","KL","AP","TS","HR"]

def plate_text(rng) -> str:
    return rng.choice(STATES)+f"{rng.randint(1,99):02d}"+''.join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ",k=2))+f"{rng.randint(0,9999):04d}"

def render_plate(text: str,rng, multiline: bool=False):
    plate=np.full((120 if multiline else 64,300,3),rng.randint(200,255),np.uint8)
    if rng.random()<0.25:
        plate[:]=(30,220,240)  # yellow commercial-style background
    cv2.rectangle(plate,(2,2),(297,plate.shape[0]-3),(30,30,30),2)
    lines=[text[:4],text[4:]] if multiline else [text]
    for i,line in enumerate(lines):
        font=rng.choice(FONTS)
        scale=min(1.25,260/cv2.getTextSize(line,font,1,2)[0][0])
        w,h=cv2.getTextSize(line,font,scale,2)[0]
        y=(i+1)*(plate.shape[0]//len(lines))-15
        cv2.putText(plate,line,((300-w)//2,y),font,scale,(15,15,15),2,cv2.LINE_AA)
    return plate

def generate(out: Path, scenes: int, chars_per_class: int,seed: int=42,alphabet: str=ALPHABET):
    if scenes<12 or chars_per_class<6:
        raise ValueError("Use at least 12 scenes and 6 samples per character.")
    if (out/"plates/train.json").exists() or (out/"characters/train.json").exists():
        raise ValueError("Output already contains a dataset. Choose a new --out directory.")
    rng=random.Random(seed)
    nrng=np.random.default_rng(seed)
    records=[]
    used=set()
    for i in range(scenes):
        image=np.full((320,480,3),rng.randint(50,150),np.uint8)
        image=np.clip(image.astype(float)+nrng.normal(0,8,image.shape),0,255).astype(np.uint8)
        boxes,texts=[],[]
        for j in range(rng.choices([0,1,2],[0.15,0.55,0.30])[0]):
            text=plate_text(rng)
            while text in used:
                text=plate_text(rng)
            used.add(text)
            multiline=rng.random()<0.25
            plate=render_plate(text,rng,multiline)
            w=rng.randint(95,190)
            h=round(w*plate.shape[0]/plate.shape[1])
            x=rng.randint(10,210-w) if j==0 else rng.randint(245,475-w)
            y=rng.randint(150,310-h)
            cv2.rectangle(image,(max(0,x-15),max(0,y-95)),(min(479,x+w+15),min(319,y+h+10)),
                          tuple(rng.randint(30,190) for _ in range(3)),-1)
            image[y:y+h,x:x+w]=cv2.resize(plate,(w,h))
            boxes.append([x,y,x+w,y+h])
            texts.append(text)
        path=out/"plates/images"/f"scene_{i:05d}.png"
        path.parent.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(path),image)
        records.append({"image":f"images/{path.name}","boxes":boxes,"texts":texts,
                        "group":f"scene-{i}","sha256":image_hash(path),"source":"procedural-synthetic"})
    write_splits(out/"plates",records,seed,"Self-generated procedural scenes; NOT real-world accuracy data")
    records=[]
    for label in alphabet:
        for i in range(chars_per_class):
            mask=np.zeros((70,70),np.uint8)
            cv2.putText(mask,label,(rng.randint(8,18),55),rng.choice(FONTS),
                        rng.uniform(1.1,1.8),255,rng.choice([1,2,3]),cv2.LINE_AA)
            m=cv2.getRotationMatrix2D((35,35),rng.uniform(-7,7),1)
            mask=cv2.warpAffine(mask,m,(70,70))
            mask=cv2.threshold(mask,80,255,cv2.THRESH_BINARY)[1]
            mask=normalize_character(mask)
            path=out/"characters/images"/f"{label}_{i:05d}.png"
            path.parent.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(path),mask)
            records.append({"image":f"images/{path.name}","label":label,
                            "group":f"glyph-{label}-{i}","sha256":image_hash(path)})
    write_splits(out/"characters",records,seed,"Synthetic OpenCV Hershey glyphs; shared font families across splits; real-domain evaluation required")
    save_json(out/"characters/alphabet.json",alphabet)
    print(f"Generated {scenes} scenes and {len(records)} glyphs in {out}")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=ROOT/"data/synthetic")
    p.add_argument("--scenes",type=int,default=300)
    p.add_argument("--chars-per-class",type=int,default=40)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--alphabet",default=ALPHABET)
    args=p.parse_args()
    if len(set(args.alphabet))!=len(args.alphabet) or not args.alphabet.isalnum():
        p.error("Alphabet must contain unique alphanumeric characters.")
    generate(args.out,args.scenes,args.chars_per_class,args.seed,args.alphabet)

if __name__=="__main__":
    main()
