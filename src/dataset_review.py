"""Validated review records. No OCR guesses or automatic training."""
import json
from pathlib import Path
import re

def validate_review(record: dict,rows: list[dict],group: str,kind: str,checked: bool) -> dict:
    if not checked:
        raise ValueError("Confirm that you checked all visible plates before saving.")
    if not group.strip():
        raise ValueError("A capture/vehicle group is required.")
    if kind not in {"Full vehicle scene","Plate close-up","Collage / article graphic","Exclude"}:
        raise ValueError("Select the image type.")
    boxes=[];texts=[];unreadable=[]
    for row in rows:
        if kind=="Exclude":
            break
        try:
            coords=[float(row[k]) for k in ("x1","y1","x2","y2")]
        except (ValueError,TypeError,KeyError) as exc:
            raise ValueError("Every box needs four numeric coordinates.") from exc
        x1,y1,x2,y2=coords
        if not(0<=x1<x2<=record["width"] and 0<=y1<y2<=record["height"]):
            raise ValueError("A box is outside the image or has zero size.")
        text=re.sub(r"[\s-]","",str(row.get("text","") or "").upper())
        unknown=bool(row.get("unreadable",False))
        if not unknown and (not text or not re.fullmatch(r"[A-Z0-9]+",text)):
            raise ValueError("Enter the actual registration, or mark the plate unreadable.")
        boxes.append(coords);texts.append("" if unknown else text);unreadable.append(unknown)
    if not boxes and kind!="Exclude":
        raise ValueError("Keep at least one plate box, or mark this image Exclude.")
    return {**record,"boxes":boxes,"texts":texts,"unreadable":unreadable,"group":group.strip(),
            "group_verified":True,"image_kind":kind,"status":"excluded" if kind=="Exclude" else "reviewed",
            "review_scope":"Plate boxes and transcription only; character boxes not yet reviewed"}

def save_review(folder: Path,record: dict) -> Path:
    identity=record["id"]
    if not re.fullmatch(r"[0-9a-f]{20}",identity):
        raise ValueError("Invalid image identity.")
    target=folder/"reviews"/(identity+".json")
    target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_suffix(".tmp")
    temp.write_text(json.dumps(record,indent=2),encoding="utf-8")
    temp.replace(target)
    return target

def load_reviews(folder: Path) -> dict:
    return {p.stem:json.loads(p.read_text(encoding="utf-8")) for p in (folder/"reviews").glob("*.json")}

def source_manifest(folder: Path,reviews: dict) -> dict:
    audit=json.loads((folder/"audit.json").read_text(encoding="utf-8"))
    records=[]
    for r in reviews.values():
        if r.get("status")=="reviewed":
            records.append({**r,"image":str((folder/r["image"]).resolve())})
    return {"source":audit["source"],"license":audit["license"],"records":records,
        "note":"Reviewed plate boxes/text only. No character annotations yet. Separate image types before detector training."}
