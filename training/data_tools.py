"""Manifest validation and leakage-resistant grouping."""
import hashlib
import json
import random
from pathlib import Path
from src.preprocessing import read_image

def save_json(path: Path,value) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2),encoding="utf-8")

def grouped_split(records: list[dict], seed: int = 42) -> dict[str,list[dict]]:
    """Union shared group IDs, image hashes, and plate identities BEFORE splitting."""
    parent=list(range(len(records)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    seen={}
    for i,r in enumerate(records):
        keys=[("group",r["group"])]
        if r.get("sha256"):
            keys.append(("hash",r["sha256"]))
        keys.extend(("plate",t.replace(" ","").upper()) for t in r.get("texts",[]) if t)
        for key in keys:
            if key in seen:
                parent[find(i)]=find(seen[key])
            else:
                seen[key]=i
    groups={}
    for i,r in enumerate(records):
        groups.setdefault(find(i),[]).append(r)
    items=list(groups.values())
    if len(items)<3:
        raise ValueError("At least three independent groups are required for train/val/test.")
    random.Random(seed).shuffle(items)
    n=len(items)
    ntrain=max(1,min(n-2,round(n*0.7)))
    nval=max(1,min(n-ntrain-1,round(n*0.15)))
    return {"train":[r for g in items[:ntrain] for r in g],
            "val":[r for g in items[ntrain:ntrain+nval] for r in g],
            "test":[r for g in items[ntrain+nval:] for r in g]}

def write_splits(out: Path,records: list[dict],seed: int,provenance: str) -> None:
    splits=grouped_split(records,seed)
    for name,rows in splits.items():
        save_json(out/f"{name}.json",rows)
    save_json(out/"dataset_card.json",{"seed":seed,"provenance":provenance,
        "counts":{k:len(v) for k,v in splits.items()},
        "split_unit":"Connected groups sharing capture ID, identical image hash or plate text",
        "note":"Ratios approximate by group; never split adjacent frames independently."})

def image_hash(path: Path) -> str:
    image=read_image(path)
    return hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()
