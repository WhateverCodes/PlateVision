"""Measure exact registration accuracy and character edit error on a named split."""
import argparse,json
from pathlib import Path
from src.runtime import configure
from src.plate_reader import Reader
from src.preprocessing import read_image
from .data_tools import save_json

def edit_distance(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]

def evaluate(data,checkpoint,split,out):
    configure(2);reader=Reader(checkpoint);rows=json.loads((Path(data)/(split+".json")).read_text());results=[]
    for r in rows:
        predicted,scores=reader.read(read_image(Path(r["image"])))
        results.append({"id":r["id"],"expected":r["text"],"predicted":predicted,"exact":predicted==r["text"],"edit_distance":edit_distance(predicted,r["text"]),"minimum_token_score":min(scores) if scores else 0})
    report={"split":split,"checkpoint":str(checkpoint),"images":len(rows),"exact_matches":sum(r["exact"] for r in results),"exact_accuracy":sum(r["exact"] for r in results)/len(rows),"character_error_rate":sum(r["edit_distance"] for r in results)/sum(len(r["expected"]) for r in results),"results":results,"note":"Greedy CTC decoding; no confidence filtering or format corrections. Small pilot split; not a broad real-world accuracy estimate."}
    save_json(Path(out),report);print(json.dumps({k:v for k,v in report.items() if k!="results"},indent=2));return report
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True);p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--split",choices=["val","test"],default="val");p.add_argument("--out",type=Path,required=True);a=p.parse_args();evaluate(a.data,a.checkpoint,a.split,a.out)
