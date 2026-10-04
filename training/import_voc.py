"""Convert Pascal VOC plate boxes to an auditable source manifest."""
import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from .data_tools import save_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations",type=Path,required=True)
    p.add_argument("--images",type=Path,required=True)
    p.add_argument("--groups",type=Path,required=True,help="CSV columns filename,group; group by vehicle/capture")
    p.add_argument("--source",required=True)
    p.add_argument("--license",required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--class-name",default="number_plate")
    a=p.parse_args()
    with a.groups.open(newline="",encoding="utf-8-sig") as f:
        groups={r["filename"]:r["group"] for r in csv.DictReader(f)}
    records=[]
    for xml in sorted(a.annotations.glob("*.xml")):
        content=xml.read_text(encoding="utf-8")
        if "<!DOCTYPE" in content or "<!ENTITY" in content:
            raise ValueError("DTD/entity declarations are not accepted.")
        root=ET.fromstring(content)
        name=root.findtext("filename")
        if not name or name not in groups:
            raise ValueError(f"Missing filename/group for {xml.name}")
        boxes=[]
        for obj in root.findall("object"):
            if obj.findtext("name")!=a.class_name:
                continue
            box=obj.find("bndbox")
            # VOC coordinates are one-based inclusive; internal coordinates are zero-based half-open.
            x1,y1,x2,y2=[int(float(box.findtext(k))) for k in ("xmin","ymin","xmax","ymax")]
            boxes.append([x1-1,y1-1,x2,y2])
        records.append({"image":str((a.images/name).resolve()),"boxes":boxes,"group":groups[name]})
    save_json(a.out,{"source":a.source,"license":a.license,"records":records})
    print(f"Wrote {len(records)} records; inspect boxes before training.")

if __name__=="__main__":
    main()
