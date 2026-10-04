"""Bounded experiment using reviewed training labels and mild glyph augmentation.

Initial weights are our own CNN, never the pretrained OCR model. Character
alignment remains provisional; evaluation records are not used for training.
"""
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from config import ROOT, ALPHABET
from src.character_classifier import Classifier
from src.runtime import configure
from training.data_tools import save_json


def train(seconds=120, epochs=4):
    configure(2)
    data = ROOT/'data/character_adaptation_v2'
    out = ROOT/'models/character_reviewed_augmented'
    if out.exists():
        raise ValueError('Preserve prior experiment; choose a new output directory.')
    audit = json.loads((data/'alignment_audit.json').read_text())
    reviewed = {r['id'] for r in audit if r['label_origin']=='human_reviewed'}
    records = json.loads((data/'train.json').read_text())
    rows = [r for r in records if r['origin']=='synthetic'
            or Path(r['image']).stem.rsplit('_',1)[0] in reviewed]
    # Synthetic replay helps preserve classes missing from the reviewed crops.
    images = np.stack([cv2.imread(r['image'],0) for r in rows])
    targets = torch.tensor([ALPHABET.index(r['label']) for r in rows])
    model = Classifier(ROOT/'models/character_adaptation_v1/last.pt').model.train()
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.00002,weight_decay=0.0001)
    rng = np.random.default_rng(123)
    weights = np.array([2.0 if r['origin']=='weak_alignment' else 1.0 for r in rows])
    weights /= weights.sum()
    start=time.monotonic();steps=0;history=[]
    out.mkdir(parents=True)
    for epoch in range(epochs):
        order=rng.choice(len(rows),size=len(rows),replace=True,p=weights)
        total=0.;count=0
        for offset in range(0,len(order),24):
            if time.monotonic()-start>=seconds:break
            chosen=order[offset:offset+24];batch=[]
            for i in chosen:
                im=images[i].copy()
                if rng.random()<0.7:
                    mat=cv2.getRotationMatrix2D((15.5,15.5),rng.uniform(-7,7),rng.uniform(.94,1.06))
                    mat[:,2]+=rng.uniform(-1,1,2)
                    im=cv2.warpAffine(im,mat,(32,32),borderValue=0)
                    if rng.random()<0.25:im=cv2.GaussianBlur(im,(3,3),0.5)
                batch.append(im)
            x=torch.from_numpy(np.stack(batch)).float()[:,None]/255
            optimizer.zero_grad();loss=nn.functional.cross_entropy(model(x),targets[chosen])
            if not torch.isfinite(loss):raise ValueError('Non-finite loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            steps+=1;total+=float(loss.detach())*len(chosen);count+=len(chosen)
            time.sleep(.04)
        history.append({'epoch':epoch+1,'samples':count,'complete':count==len(order),'loss':total/max(count,1)})
        if count<len(order):break
    provenance='Own CNN weights; reviewed training plate labels with automatic glyph alignment, synthetic replay and mild augmentation. No pretrained OCR weights or evaluation images used.'
    torch.save({'kind':'character_classifier','format_version':1,'alphabet':ALPHABET,
                'model':model.state_dict(),'trained_steps':steps,'provenance':provenance},out/'last.pt')
    report={'history':history,'seconds':time.monotonic()-start,'training_glyphs':len(rows),
            'reviewed_plate_groups':len(reviewed),'provenance':provenance}
    save_json(out/'report.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':train()
