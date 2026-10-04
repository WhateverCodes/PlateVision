"""Single-class AP at IoU .50:.95 and precision/recall at a declared operating score."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torchvision.ops import box_iou
from config import DEFAULTS,ROOT
from src.runtime import configure
from src.plate_detector import Detector
from src.preprocessing import read_image
from src.visualization import annotate,jpeg
from .data_tools import save_json

def metrics(predictions,truths,iou_threshold=0.5,score_threshold=0.5):
    ranked=sorted([(float(p["score"]),i,p["box"]) for i,ps in enumerate(predictions) for p in ps],reverse=True)
    used={i:set() for i in range(len(truths))}
    tp,fp,scores,matched_ious=[],[],[],[]
    total=sum(len(g) for g in truths)
    for score,i,box in ranked:
        gt=truths[i]
        match=False
        if gt:
            values=box_iou(torch.tensor([box],dtype=torch.float32),torch.tensor(gt,dtype=torch.float32))[0].numpy()
            values[list(used[i])]=-1
            j=int(values.argmax())
            if values[j]>=iou_threshold:
                used[i].add(j)
                match=True
                matched_ious.append(float(values[j]))
        tp.append(int(match)); fp.append(int(not match)); scores.append(score)
    tp,fp=np.array(tp),np.array(fp)
    cum_tp,cum_fp=np.cumsum(tp),np.cumsum(fp)
    recall=cum_tp/max(total,1)
    precision=cum_tp/np.maximum(cum_tp+cum_fp,1)
    ap=float(np.mean([precision[recall>=r].max() if np.any(recall>=r) else 0 for r in np.linspace(0,1,101)])) if total else None
    active=np.array(scores)>=score_threshold
    count_tp=int(tp[active].sum()); count_fp=int(fp[active].sum())
    return {"ap":ap,"precision":count_tp/max(count_tp+count_fp,1),"recall":count_tp/max(total,1),
            "true_positives":count_tp,"false_positives":count_fp,"ground_truth_plates":total,
            "mean_matched_iou":float(np.mean(matched_ious)) if matched_ious else 0.,
            "curve_precision":precision.tolist(),"curve_recall":recall.tolist()}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--weights",type=Path,default=DEFAULTS.detector_path)
    p.add_argument("--out",type=Path,default=ROOT/"outputs/evaluation/detector")
    p.add_argument("--threshold",type=float,default=0.5)
    p.add_argument("--split",choices=["val","test"],default="test")
    a=p.parse_args()
    configure()
    records=json.loads((a.data/(a.split+".json")).read_text())
    detector=Detector(a.weights,threshold=0.01)
    predictions=[]; truths=[]
    a.out.mkdir(parents=True,exist_ok=True)
    for i,r in enumerate(records):
        image=read_image(a.data/r["image"])
        result=detector(image)
        predictions.append(result); truths.append(r["boxes"])
        if i<6:
            (a.out/f"sample_{i}.jpg").write_bytes(jpeg(annotate(image,[v for v in result if v["score"]>=a.threshold])))
    results=[metrics(predictions,truths,float(t),a.threshold) for t in np.arange(0.5,0.96,0.05)]
    summary={"split":a.split,"AP50":results[0]["ap"],"mAP50_95":float(np.mean([r["ap"] for r in results])) if any(truths) else None,
             "operating_score":a.threshold,"candidate_score_floor":0.01,"provenance":detector.provenance,**results[0]}
    save_json(a.out/"metrics.json",summary)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    plt.plot(results[0]["curve_recall"],results[0]["curve_precision"])
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Plate detection, IoU 0.50")
    plt.savefig(a.out/"precision_recall.png"); plt.close()
    print(json.dumps({k:v for k,v in summary.items() if not k.startswith("curve")},indent=2))

if __name__=="__main__":
    main()
