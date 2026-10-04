"""Safely import a YOLO image/label ZIP into a deduplicated REVIEW collection."""
import argparse
from collections import defaultdict,Counter
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path,PurePosixPath
import zipfile
import cv2
import numpy as np
from PIL import Image
from src.preprocessing import read_image
from .data_tools import save_json

SOURCE="https://www.kaggle.com/datasets/kedarsai/indian-license-plates-with-labels"
LICENSE_NOTE="Kaggle listing declares CC0; source mixes web-collected images and model-generated boxes. Upstream image rights not independently verified."

def yolo_boxes(text: str,width: int,height: int) -> list[list[float]]:
    boxes=[]
    for line in text.strip().splitlines():
        values=[float(v) for v in line.split()]
        if len(values)!=5 or not all(math.isfinite(v) for v in values):
            raise ValueError("Each label row must contain five finite numbers.")
        label,cx,cy,w,h=values
        if label!=0 or not(0<=cx<=1 and 0<=cy<=1 and 0<w<=1 and 0<h<=1):
            raise ValueError("Unsupported class or invalid normalized coordinates.")
        box=[cx-w/2,cy-h/2,cx+w/2,cy+h/2]
        if min(box)<-0.00001 or max(box)>1.00001:
            raise ValueError("Bounding box extends beyond the image.")
        boxes.append([round(max(0,min(1,v))*scale,3) for v,scale in zip(box,[width,height,width,height])])
    return sorted(boxes)

def safe_members(z: zipfile.ZipFile):
    infos=[i for i in z.infolist() if not i.is_dir()]
    if len(infos)>10000 or sum(i.file_size for i in infos)>1_000_000_000:
        raise ValueError("Archive exceeds import limits.")
    names=set()
    for info in infos:
        path=PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in info.filename or ":" in info.filename:
            raise ValueError("Unsafe archive path.")
        if info.filename in names:
            raise ValueError("Repeated archive member name.")
        if info.file_size>25_000_000:
            raise ValueError("Individual archive file exceeds 25 MB.")
        names.add(info.filename)
    return infos

def import_dataset(archive: Path,out: Path) -> dict:
    cv2.setNumThreads(1)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output already contains files. Choose a new output folder.")
    grouped=defaultdict(list); rejected=[]; fingerprints={}
    with zipfile.ZipFile(archive) as z:
        infos=safe_members(z)
        image_names=sorted(i.filename for i in infos if PurePosixPath(i.filename).parts[0]=="images"
                           and Path(i.filename).suffix.lower() in {".jpg",".jpeg",".png",".webp"})
        labels={}
        for i in infos:
            if PurePosixPath(i.filename).parts[0]=="labels" and i.filename.lower().endswith(".txt"):
                stem=Path(i.filename).stem
                if stem in labels:
                    raise ValueError("Ambiguous repeated label stem.")
                labels[stem]=i.filename
        counts=Counter()
        for index,name in enumerate(image_names):
            try:
                raw=z.read(name)
                with Image.open(BytesIO(raw)) as im:
                    if im.getexif().get(274,1)!=1:
                        raise ValueError("EXIF rotation requires manual coordinate review before import.")
                image=read_image(raw)
                h,w=image.shape[:2]
                digest=hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()
                label_path=labels.get(Path(name).stem)
                boxes=None; reason=None
                if label_path is None:
                    counts["missing_label_files"]+=1; reason="No label file"
                else:
                    try:
                        boxes=yolo_boxes(z.read(label_path).decode("utf-8-sig"),w,h)
                        if not boxes:
                            counts["empty_label_files"]+=1; reason="Empty label is not assumed to mean a true negative"
                    except ValueError as e:
                        counts["invalid_label_files"]+=1; reason=str(e)
                grouped[digest].append({"original":name,"boxes":boxes,"reason":reason})
                if digest not in fingerprints:
                    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                    small=cv2.resize(gray,(9,8))
                    dhash=sum(int(v)<<k for k,v in enumerate((small[:,1:]>small[:,:-1]).ravel()))
                    fingerprints[digest]=(dhash,w/h,cv2.resize(gray,(32,32)))
                    target=out/"images"/(digest[:20]+".png")
                    target.parent.mkdir(parents=True,exist_ok=True)
                    ok,encoded=cv2.imencode(".png",image)
                    if not ok: raise ValueError("Could not save decoded image.")
                    target.write_bytes(encoded.tobytes())
            except Exception as e:
                rejected.append({"original":name,"reason":str(e)})
            if (index+1)%500==0:
                print(f"Inspected {index+1}/{len(image_names)} source images",flush=True)
        records=[]
        for digest,variants in grouped.items():
            candidates=[v["boxes"] for v in variants if v["boxes"]]
            consistent=bool(candidates) and all(len(b)==len(candidates[0]) and
                np.allclose(b,candidates[0],atol=1,rtol=0) for b in candidates)
            if not consistent:
                status="quarantine"; boxes=[]
                reason="No usable label" if not candidates else "Conflicting labels on identical images"
            else:
                status="pending_review"; boxes=candidates[0]; reason=""
            with Image.open(out/"images"/(digest[:20]+".png")) as im:
                w,h=im.size
            coverage=max(((b[2]-b[0])*(b[3]-b[1])/(w*h) for b in boxes),default=0)
            hint="plate_closeup" if coverage>=0.60 and len(boxes)==1 else "scene_or_collage"
            records.append({"id":digest[:20],"image":f"images/{digest[:20]}.png","sha256":digest,
                "width":w,"height":h,"original_files":[v["original"] for v in variants],
                "boxes":boxes,"status":status,"reason":reason,"suggested_kind":hint,
                "group":digest[:20],"group_verified":False,"texts":[""]*len(boxes)})
    # Conservative visual grouping only; actual repeated registrations still need review.
    parents=list(range(len(records)))
    def find(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]];i=parents[i]
        return i
    similar=[]
    for i,r in enumerate(records):
        ha,aspect,a=fingerprints[r["sha256"]]
        for j in range(i):
            hb,other,b=fingerprints[records[j]["sha256"]]
            if abs(math.log(aspect/other))<0.08 and (ha^hb).bit_count()<=4 and np.abs(a.astype(float)-b).mean()<18:
                parents[find(i)]=find(j)
                similar.append([r["id"],records[j]["id"]])
        r["group"]=records[find(i)]["id"]
    for i,r in enumerate(records):
        r["group"]="visual-"+records[find(i)]["id"]
    ready=[r for r in records if r["status"]=="pending_review"]
    report={"archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
        "source":SOURCE,"license":LICENSE_NOTE,"source_images":len(image_names),"source_label_files":len(labels),
        **dict(counts),"unique_decoded_images":len(records),"duplicate_copies_removed":sum(len(v)-1 for v in grouped.values()),
        "images_with_usable_boxes":len(ready),"quarantined_unique_images":len(records)-len(ready),
        "usable_plate_boxes":sum(len(r["boxes"]) for r in ready),
        "geometry_hints":dict(Counter(r["suggested_kind"] for r in ready)),
        "near_duplicate_candidate_pairs":len(similar),"rejected":rejected,
        "training_started":False,"split_created":False,
        "limitations":["Geometry hints are not manually verified scene categories.",
            "Source labels may be model-generated. Review boxes and missing plates.",
            "Capture identities are not supplied. Visual grouping cannot establish leak-free splits.",
            "No registration transcriptions or character labels were supplied."]}
    save_json(out/"catalog.json",records)
    save_json(out/"audit.json",report)
    save_json(out/"near_duplicate_candidates.json",similar)
    print(json.dumps({k:v for k,v in report.items() if k not in ("rejected","limitations")},indent=2))
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    import_dataset(a.archive,a.out)

if __name__=="__main__":
    main()
