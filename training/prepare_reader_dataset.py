"""Freeze reviewed labels and optional source-labeled training data into grouped splits."""
import argparse,json,hashlib
from pathlib import Path
from collections import Counter
import cv2
from src.dataset_review import load_reviews
from src.preprocessing import read_image,grayscale
from .data_tools import grouped_split,save_json

def prepare(folder,out,holdout=None):
    folder,out=Path(folder),Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError("Use a new output directory; this snapshot is immutable.")
    reviews=load_reviews(folder);catalog=json.loads((folder/"catalog.json").read_text())
    reserved=json.loads(Path(holdout).read_text())["records"] if holdout else []
    reserved_texts={t for r in reserved for t in r["texts"]};reserved_hashes={r["sha256"] for r in reserved}
    rows=[];skipped=Counter();thumbs=[];dhash=[]
    for source in catalog:
        r=reviews.get(source["id"],source)
        if r["status"].startswith("excluded"):skipped["excluded"]+=1;continue
        if r["status"]!="reviewed" and r["flags"]:skipped["awaiting_flag_review"]+=1;continue
        if r["text"] in reserved_texts or r["sha256"] in reserved_hashes:skipped["reserved_test_overlap"]+=1;continue
        path=(folder/r["image"]).resolve();image=read_image(path);gray=grayscale(image)
        small=cv2.resize(gray,(9,8));bits=(small[:,1:]>small[:,:-1]).reshape(-1)
        dhash.append(sum(int(b)<<i for i,b in enumerate(bits)));thumbs.append(cv2.resize(gray,(32,16)))
        rows.append({**r,"image":str(path),"texts":[r["text"]],"label_origin":"human_reviewed" if r["status"]=="reviewed" else "source_unverified","file_sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):parent[find(a)]=find(b)
    seen={};near_pairs=0
    for i,r in enumerate(rows):
        for key in [("source",r["group"]),("text",r["text"]),("pixels",r["sha256"])]:
            if key in seen:union(i,seen[key])
            else:seen[key]=i
        for j in range(i):
            if (dhash[i]^dhash[j]).bit_count()<=3 and abs(r["width"]/r["height"]/(rows[j]["width"]/rows[j]["height"])-1)<0.12:
                if abs(thumbs[i].astype(float)-thumbs[j].astype(float)).mean()<15:union(i,j);near_pairs+=1
    for i,r in enumerate(rows):r["group"]="reader-group-"+rows[find(i)]["id"]
    splits=grouped_split(rows)
    withheld=[]
    for name in ("val","test"):
        withheld.extend(r for r in splits[name] if r["label_origin"]!="human_reviewed")
        splits[name]=[r for r in splits[name] if r["label_origin"]=="human_reviewed"]
    if any(not rows for rows in splits.values()):raise ValueError("Need more independently reviewed groups for nonempty evaluation splits.")
    for name,items in splits.items():save_json(out/(name+".json"),items)
    save_json(out/"withheld_unverified.json",withheld)
    card={"counts":{k:len(v) for k,v in splits.items()},"label_origins":{k:dict(Counter(r["label_origin"] for r in v)) for k,v in splits.items()},"skipped":dict(skipped),"withheld_unverified":len(withheld),"near_duplicate_pairs_grouped":near_pairs,"provenance":"KP00011 plate crops with preserved source labels and local human corrections; no pretrained reader weights","note":"Training includes source-unverified labels. Validation/test contain human-reviewed labels only. Small evaluation sets are preliminary. Grouping reduces leakage but cannot certify independent capture origins. External test photos remain reserved. Two-line plates may need additional preprocessing.","source_audit":json.loads((folder/"audit.json").read_text())}
    save_json(out/"dataset_card.json",card)
    return card
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--folder",type=Path,required=True);p.add_argument("--out",type=Path,required=True);p.add_argument("--holdout",type=Path);a=p.parse_args();print(json.dumps(prepare(a.folder,a.out,a.holdout),indent=2))
