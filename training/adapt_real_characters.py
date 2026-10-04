"""Experimental weakly aligned real-character adaptation; never uses validation/test labels."""
import argparse,json,time,random
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
from config import ALPHABET
from src.runtime import configure
from src.character_classifier import Classifier
from src.character_segmenter import segmentation_candidates
from src.preprocessing import read_image
from .data_tools import save_json

def prepare(data,out,base,synthetic):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError("Output exists; preserve prior annotations.")
    configure(2);classifier=Classifier(base);rows=[];plates=[]
    for r in json.loads((Path(data)/"train.json").read_text()):
        choices=[]
        for _,chars,mask in segmentation_candidates(read_image(Path(r["image"]))):
            if len(chars)!=len(r["text"]):continue
            predicted,scores=classifier(chars)
            agreement=sum(a==b for a,b in zip(predicted,r["text"]))/len(chars)
            if agreement>=.7:choices.append((agreement,float(np.mean(scores)),chars,predicted))
        if not choices:continue
        agreement,score,chars,predicted=max(choices,key=lambda c:c[:2])
        plates.append({"id":r["id"],"text":r["text"],"predicted":predicted,"agreement":agreement,"label_origin":r["label_origin"]})
        for i,(char,label) in enumerate(zip(chars,r["text"])):
            path=out/"images"/(r["id"]+f"_{i}.png");path.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(path),char.image)
            rows.append({"image":str(path.resolve()),"label":label,"group":r["group"],"origin":"weak_alignment"})
    for r in json.loads((Path(synthetic)/"train.json").read_text()):
        rows.append({**r,"image":str((Path(synthetic)/r["image"]).resolve()),"origin":"synthetic"})
    save_json(out/"train.json",rows);save_json(out/"alignment_audit.json",plates)
    card={"accepted_training_plates":len(plates),"weak_real_characters":sum(r["origin"]=="weak_alignment" for r in rows),"synthetic_characters":sum(r["origin"]=="synthetic" for r in rows),"note":"Experimental automatic alignment, not human character annotations. Count agreement and >=70% classifier/text agreement do not guarantee correct boxes. Only reader training split used. Validation and both test sets unused."}
    save_json(out/"dataset_card.json",card);print(json.dumps(card));return card

def train(data,out,base,seconds=60,epochs=3):
    configure(2);data,out=Path(data),Path(out)
    if (out/"last.pt").exists():raise ValueError("Use a fresh experiment directory.")
    classifier=Classifier(base);model=classifier.model;model.train()
    rows=json.loads((data/"train.json").read_text());optimizer=torch.optim.AdamW(model.parameters(),lr=.00003,weight_decay=.0001)
    out.mkdir(parents=True,exist_ok=True);start=time.monotonic();steps=0;history=[]
    for epoch in range(epochs):
        order=list(range(len(rows)));random.Random(42+epoch).shuffle(order);total=0.;count=0
        for offset in range(0,len(order),24):
            if time.monotonic()-start>=seconds:break
            batch=[rows[i] for i in order[offset:offset+24]]
            images=np.stack([cv2.imread(r["image"],cv2.IMREAD_GRAYSCALE) for r in batch])
            x=torch.from_numpy(images).float()[:,None]/255;y=torch.tensor([ALPHABET.index(r["label"]) for r in batch])
            optimizer.zero_grad();loss=nn.functional.cross_entropy(model(x),y)
            if not torch.isfinite(loss):raise ValueError("Non-finite adaptation loss.")
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();steps+=1;total+=float(loss.detach())*len(batch);count+=len(batch);time.sleep(.05)
        history.append({"epoch":epoch+1,"samples_seen":count,"loss":total/max(1,count),"complete":count==len(rows)})
        if time.monotonic()-start>=seconds:break
    checkpoint={"kind":"character_classifier","format_version":1,"alphabet":ALPHABET,"model":model.state_dict(),"trained_steps":steps,"provenance":"Experimental adaptation of own synthetic CNN using weakly aligned real training crops plus synthetic replay; "+str(data.resolve()),"adaptation_history":history}
    torch.save(checkpoint,out/"last.pt");save_json(out/"history.json",history)
    print(json.dumps({"steps":steps,"seconds":time.monotonic()-start,"history":history}));return history
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["prepare","train"]);p.add_argument("--data",type=Path,required=True);p.add_argument("--out",type=Path,required=True);p.add_argument("--base",type=Path,default=Path("models/character_classifier/best.pt"));p.add_argument("--synthetic",type=Path,default=Path("data/synthetic/characters"));p.add_argument("--seconds",type=float,default=60);a=p.parse_args()
    if a.mode=="prepare":prepare(a.data,a.out,a.base,a.synthetic)
    else:train(a.data,a.out,a.base,a.seconds)
