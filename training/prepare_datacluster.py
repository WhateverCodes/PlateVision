"""Prepare the small DataCluster development sample with orientation-correct boxes."""
import argparse,io,json,hashlib,re
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET
from PIL import Image,ImageOps
from .data_tools import write_splits

def prepare(archive,out):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError("Output already exists.")
    rows=[]
    with ZipFile(archive) as z:
        names={Path(n).name:n for n in z.namelist() if n.endswith(".jpg")}
        for n in z.namelist():
            if not n.endswith(".xml"):continue
            node=ET.fromstring(z.read(n));name=node.findtext("filename")
            image=ImageOps.exif_transpose(Image.open(io.BytesIO(z.read(names[name])))).convert("RGB");w,h=image.size
            if (w,h)!=(int(node.findtext("size/width")),int(node.findtext("size/height"))):raise ValueError("Annotation orientation mismatch")
            digest=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest();scale=min(1,1280/max(w,h));nw,nh=round(w*scale),round(h*scale)
            image=image.resize((nw,nh));boxes=[];texts=[]
            for obj in node.findall("object"):
                box=[float(obj.findtext("bndbox/"+k)) for k in ("xmin","ymin","xmax","ymax")]
                if not (0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h):raise ValueError("Invalid plate box")
                boxes.append([box[0]*nw/w,box[1]*nh/h,box[2]*nw/w,box[3]*nh/h]);text=""
                for attr in obj.findall("attributes/attribute"):
                    if attr.findtext("name")=="number_plate_text":text=re.sub("[^A-Z0-9]","",attr.findtext("value","").upper())
                texts.append(text)
            dest=out/"images"/(digest[:20]+".png");dest.parent.mkdir(parents=True,exist_ok=True);image.save(dest)
            rows.append({"image":"images/"+dest.name,"group":digest,"sha256":digest,"boxes":boxes,"texts":texts,"source_file":name})
    write_splits(out,rows,42,"DataCluster supplied 47-photo development sample; EXIF normalized; source XML boxes, not manually verified. Text/hash grouping only; independent captures unverified. This sample was already used to inspect classical proposals, so splits are development-only.")
    print((out/"dataset_card.json").read_text())
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--archive",required=True);p.add_argument("--out",required=True);a=p.parse_args();prepare(a.archive,a.out)
