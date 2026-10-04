"""Compact randomly initialized CNN/GRU whole-plate reader with CTC decoding."""
import cv2
import numpy as np
import torch
from torch import nn
from config import ALPHABET
from .preprocessing import grayscale

HEIGHT,WIDTH=32,192
class PlateReader(nn.Module):
    def __init__(self,architecture=2):
        super().__init__()
        self.architecture=architecture
        layers=[];previous=1
        for channels,pool in [(16,(2,2)),(32,(2,2)),(64,(2,1)),(64,(2,1)),(64,(2,1))]:
            layers.append(nn.Conv2d(previous,channels,3,padding=1))
            if architecture==2:layers.append(nn.BatchNorm2d(channels))
            layers.extend([nn.ReLU(),nn.MaxPool2d(pool)])
            previous=channels
        self.features=nn.Sequential(*layers)
        self.sequence=nn.GRU(64,64,batch_first=True,bidirectional=True)
        self.head=nn.Linear(128,len(ALPHABET)+1)
    def forward(self,x):
        features=self.features(x).squeeze(2).transpose(1,2)
        values,_=self.sequence(features)
        return self.head(values).log_softmax(-1).transpose(0,1)

def prepare_crop(image):
    gray=grayscale(image)
    resized=cv2.resize(gray,(WIDTH,HEIGHT),interpolation=cv2.INTER_AREA)
    return torch.from_numpy(np.ascontiguousarray(resized)).float().unsqueeze(0)/127.5-1

def decode(log_probs):
    """Collapse repeated tokens, with blank=0; blank separates repeated letters."""
    ids=log_probs.argmax(-1).transpose(0,1).tolist()
    texts=[]
    for sequence in ids:
        last=0;letters=[]
        for value in sequence:
            if value and value!=last:letters.append(ALPHABET[value-1])
            last=value
        texts.append("".join(letters))
    return texts


class Reader:
    def __init__(self,path):
        checkpoint=torch.load(path,map_location="cpu",weights_only=True)
        if checkpoint.get("kind")!="plate_reader" or checkpoint.get("alphabet")!=ALPHABET or not checkpoint.get("trained_steps"):
            raise ValueError("Missing or incompatible trained plate reader.")
        self.model=PlateReader(checkpoint.get("architecture",1)).eval();self.model.load_state_dict(checkpoint["model"])
        self.provenance=checkpoint.get("provenance","Whole-plate reader")
    @torch.inference_mode()
    def read(self,image):
        values=self.model(prepare_crop(image).unsqueeze(0));text=decode(values)[0]
        probs,ids=values.exp().max(-1);scores=[];last=0
        for token,score in zip(ids[:,0].tolist(),probs[:,0].tolist()):
            if token and token!=last:scores.append(score)
            last=token
        return text,scores
