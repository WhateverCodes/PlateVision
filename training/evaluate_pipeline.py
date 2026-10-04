"""End-to-end plate exact match, counting missed detections and failed segmentation."""
import argparse
import json
from dataclasses import replace
from pathlib import Path
import torch
from torchvision.ops import box_iou
from config import DEFAULTS,ROOT
from src.runtime import configure
from src.pipeline import Pipeline
from src.preprocessing import read_image
from src.validator import normalize
from .data_tools import save_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--detector",type=Path,default=DEFAULTS.detector_path)
    p.add_argument("--characters",type=Path,default=DEFAULTS.character_path)
    p.add_argument("--out",type=Path,default=ROOT/"outputs/evaluation/end_to_end")
    a=p.parse_args()
    configure()
    pipe=Pipeline(replace(DEFAULTS,detector_path=a.detector,character_path=a.characters))
    records=json.loads((a.data/"test.json").read_text())
    total=correct=false_positive=0
    failures=[]
    for record in records:
        if len(record.get("texts",[]))!=len(record["boxes"]):
            raise ValueError("End-to-end evaluation requires a ground-truth text for every plate.")
        detections=sorted(pipe(read_image(a.data/record["image"])),key=lambda d:d["score"],reverse=True)
        used=set()
        for d in detections:
            if not record["boxes"]:
                false_positive+=1
                continue
            overlaps=box_iou(torch.tensor([d["box"]],dtype=torch.float32),torch.tensor(record["boxes"],dtype=torch.float32))[0]
            if used:
                overlaps[list(used)]=-1
            j=int(overlaps.argmax())
            if overlaps[j]>=0.5:
                used.add(j)
                match=d["recognized"] and normalize(d["text"])==normalize(record["texts"][j])
                correct+=int(match)
                if not match:
                    failures.append({"image":record["image"],"expected":record["texts"][j],"predicted":d["text"]})
            else:
                false_positive+=1
        total+=len(record["boxes"])
        for j,text in enumerate(record["texts"]):
            if j not in used:
                failures.append({"image":record["image"],"expected":text,"predicted":"Missed detection"})
    summary={"ground_truth_plates":total,"exact_matches":correct,"exact_match_rate":correct/total if total else None,
             "false_positive_detections":false_positive,"failures":failures,
             "note":"All ground-truth plates counted, including detection/segmentation failures."}
    save_json(a.out/"metrics.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k!="failures"},indent=2))

if __name__=="__main__":
    main()
