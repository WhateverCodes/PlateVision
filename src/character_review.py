"""Human-checked character boxes in original image coordinates."""
import hashlib
import json
import math
from pathlib import Path

def fingerprint(record, plate):
    payload={k:record.get(k) for k in ("id","sha256","group","image")}
    payload.update(box=record["boxes"][plate],text=record["texts"][plate])
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()

def crop_bounds(record, plate):
    x1,y1,x2,y2=record["boxes"][plate]
    return math.floor(x1),math.floor(y1),math.ceil(x2),math.ceil(y2)

def save_characters(folder, record, plate, entries):
    text=record["texts"][plate]
    x0,y0,xend,yend=crop_bounds(record,plate)
    if len(entries)!=len(text) or not entries:
        raise ValueError("Check every character first.")
    chars=[]
    for letter,entry in zip(text,entries):
        if not entry or not entry.get("checked") or entry.get("label")!=letter:
            raise ValueError("Check every character first.")
        x1,y1,x2,y2=entry["box"]
        if not all(isinstance(v,int) for v in (x1,y1,x2,y2)) or not (0<=x1<x2<=xend-x0 and 0<=y1<y2<=yend-y0):
            raise ValueError("Each character needs a nonempty box inside the plate crop.")
        if entry.get("polarity") not in ("dark","light"):
            raise ValueError("Choose the character colour.")
        chars.append({"label":letter,"box":[x0+x1,y0+y1,x0+x2,y0+y2],"polarity":entry["polarity"]})
    result={"source_fingerprint":fingerprint(record,plate),"source_id":record["id"],
            "plate_index":plate,"text":text,"group":record["group"],"image":record["image"],
            "characters":chars,"status":"human_reviewed","coordinate_space":"original_image"}
    target=Path(folder)/"character_reviews"/(record["id"]+f"_{plate}.json")
    target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_suffix(".tmp")
    temp.write_text(json.dumps(result,indent=2),encoding="utf-8")
    temp.replace(target)
    return target
