"""Evaluate combined localization and reading on development data only."""
import argparse,json,time
from pathlib import Path
from dataclasses import replace
from config import DEFAULTS
from src.pipeline import Pipeline
from src.preprocessing import read_image
from src.runtime import configure
from .data_tools import save_json

def evaluate(data,weights,mode,out):
    configure(2);cfg=replace(DEFAULTS,detector_path=Path(weights),character_path=DEFAULTS.adapted_character_path,localization_mode=mode)
    pipeline=Pipeline(cfg);rows=json.loads((Path(data)/"val.json").read_text());results=[]
    def iou(a,b):
        area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
        return area/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)
    for r in rows:
        started=time.monotonic();detections=pipeline(read_image(Path(r["image"])));accepted=[]
        for d in detections:
            if not d["recognized"]:continue
            overlaps=[iou(d["box"],b) for b in r["boxes"]];j=max(range(len(overlaps)),key=overlaps.__getitem__) if overlaps else None
            expected=r["texts"][j] if j is not None and overlaps[j]>=.5 else None
            outcome="unverified_label" if expected=="" else "correct" if expected==d["text"] else "incorrect"
            accepted.append({"text":d["text"],"box":d["box"],"confidence":d["confidence"],"expected":expected,"outcome":outcome})
        results.append({"image":r["source_file"],"seconds":time.monotonic()-started,"candidates":len(detections),"accepted":accepted,"expected":r["texts"]})
    report={"mode":mode,"weights":str(weights),"images":len(rows),"correct_accepted":sum(a["outcome"]=="correct" for r in results for a in r["accepted"]),"incorrect_accepted":sum(a["outcome"]=="incorrect" for r in results for a in r["accepted"]),"unverified_accepted":sum(a["outcome"]=="unverified_label" for r in results for a in r["accepted"]),"results":results,"note":"Development-only test, exact text plus IoU >=0.5 matching. Source annotations may be incomplete. Not final independent test accuracy."}
    save_json(Path(out),report);print(json.dumps({k:v for k,v in report.items() if k!="results"},indent=2));return report
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",default="data/detector_real_v2");p.add_argument("--weights",default="models/plate_detector/best.pt");p.add_argument("--mode",choices=["learned","classical"],default="learned");p.add_argument("--out",required=True);a=p.parse_args();evaluate(a.data,a.weights,a.mode,a.out)
