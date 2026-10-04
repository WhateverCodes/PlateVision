"""Development check of proposal overlap on the 47-photo DataCluster sample."""
import io,json,time
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET
from PIL import Image,ImageOps
import cv2
import numpy as np
from src.classical_detector import ClassicalDetector
from .data_tools import save_json

def check(archive,out):
    cv2.setNumThreads(1);detector=ClassicalDetector();results=[]
    def iou(a,b):
        inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
        return inter/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)
    with ZipFile(archive) as z:
        names={Path(n).name:n for n in z.namelist() if n.endswith(".jpg")}
        for name in z.namelist():
            if not name.endswith(".xml"):continue
            node=ET.fromstring(z.read(name));image_name=node.findtext("filename")
            image=np.array(ImageOps.exif_transpose(Image.open(io.BytesIO(z.read(names[image_name])))).convert("RGB"))
            t=time.monotonic();proposals=detector(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY));seconds=time.monotonic()-t
            overlaps=[]
            for obj in node.findall("object"):
                box=[float(obj.findtext("bndbox/"+k)) for k in ("xmin","ymin","xmax","ymax")]
                overlaps.append(max([iou(box,p["box"]) for p in proposals],default=0))
            results.append({"image":image_name,"candidates":len(proposals),"seconds":seconds,"best_ious":overlaps})
    count=sum(len(r["best_ious"]) for r in results);matched=sum(v>=.5 for r in results for v in r["best_ious"])
    report={"images":len(results),"annotated_plates":count,"plates_with_proposal_iou50":matched,"proposal_recall_iou50":matched/count,"mean_candidates":sum(r["candidates"] for r in results)/len(results),"note":"Development sample only; measures candidate coverage, not recognition or detection precision. External test photos were not used.","results":results}
    save_json(Path(out),report);print(json.dumps({k:v for k,v in report.items() if k!="results"},indent=2))
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--archive",required=True);p.add_argument("--out",required=True);a=p.parse_args();check(a.archive,a.out)
