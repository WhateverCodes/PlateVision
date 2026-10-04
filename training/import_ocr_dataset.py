"""Import plate crops and text; retain source labels and defer evaluation splitting."""
import argparse,hashlib,io,json,re
from collections import Counter,defaultdict
from pathlib import Path,PurePosixPath
from zipfile import ZipFile
from PIL import Image,ImageOps

SOURCE="https://www.kaggle.com/datasets/kp00011/indian-vehical-number-plate-ocr-labeled-dataset"
def import_dataset(archive,out):
    archive,out=Path(archive),Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError("Output already contains files; existing reviews will not be overwritten.")
    rows=[];seen_paths=set();groups=defaultdict(set)
    with ZipFile(archive) as z:
        infos=z.infolist()
        if len(infos)>10000 or sum(i.file_size for i in infos)>1_000_000_000:raise ValueError("Archive exceeds import limits.")
        for info in infos:
            p=PurePosixPath(info.filename)
            if p.is_absolute() or ".." in p.parts or ":" in info.filename or "\\" in info.filename or info.file_size>25_000_000:raise ValueError("Unsafe archive entry.")
            if info.filename in seen_paths:raise ValueError("Duplicate archive path.")
            seen_paths.add(info.filename)
        for split in ("train","valid","test"):
            for line in z.read(split+"_labels.txt").decode("utf-8-sig").splitlines():
                if not line.strip():continue
                parts=line.split("\t")
                if len(parts)!=2:raise ValueError("Expected filename and text separated by a tab.")
                name,text=parts
                if PurePosixPath(name).name!=name or not name.lower().endswith(".jpg"):raise ValueError("Invalid image filename.")
                member=f"images/{split}/{name}"
                if any(r["original_file"]==member for r in rows):raise ValueError("Duplicate label row.")
                data=z.read(member)
                with Image.open(io.BytesIO(data)) as source:
                    image=ImageOps.exif_transpose(source).convert("RGB")
                w,h=image.size
                digest=hashlib.sha256(str((w,h)).encode()+image.tobytes()).hexdigest()
                flags=[]
                excluded=text.lower() in {"blur","devanagri"}
                if excluded:flags.append("Source label is a placeholder or unsupported script.")
                elif not re.fullmatch("[A-Z0-9]{8,11}",text):flags.append("Unusual text format: check against the image.")
                if h<24:flags.append("Small image: check whether every character is readable.")
                if re.fullmatch("[A-Z]{2}[0-9]{6,}",text):flags.append("Digits-only suffix: check for a letter mistaken for a digit.")
                identity=hashlib.sha256(member.encode()+data).hexdigest()[:20]
                path=out/"images"/(identity+".png");path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
                rows.append({"id":identity,"image":"images/"+path.name,"sha256":digest,"original_file":member,"original_split":split,"source_text":text,"text":"" if excluded else text,"width":w,"height":h,"flags":flags,"status":"excluded_source_placeholder" if excluded else "pending_review","group":"source-text-"+hashlib.sha256(text.encode()).hexdigest()[:16],"group_verified":False})
                groups[text].add(split)
    rows.sort(key=lambda r:(r["status"].startswith("excluded"),not bool(r["flags"]),r["original_file"]))
    audit={"source":SOURCE,"license":"Not verified; no license file in the supplied archive. Do not redistribute images without checking source terms.","archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),"images":len(rows),"reviewable":sum(r["status"]=="pending_review" for r in rows),"flagged":sum(bool(r["flags"]) and r["status"]=="pending_review" for r in rows),"excluded_placeholders":sum(r["status"]!="pending_review" for r in rows),"source_texts_across_splits":sum(len(s)>1 for s in groups.values()),"source_split_counts":dict(Counter(r["original_split"] for r in rows)),"split_created":False,"training_started":False,"note":"All original splits retained as metadata only. Final groups must combine corrected identities, image duplicates and related captures before splitting. Flags are heuristics, not an exhaustive label audit."}
    (out/"catalog.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    (out/"audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    return audit
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--archive",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    print(json.dumps(import_dataset(a.archive,a.out),indent=2))
