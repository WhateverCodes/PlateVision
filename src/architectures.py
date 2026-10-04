"""Two randomly initialized educational neural networks. No external weights."""
import math
import torch
from torch import nn
from torch.nn import functional as F

def block(a: int, b: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(a,b,3,stride,1,bias=False), nn.BatchNorm2d(b), nn.SiLU())

class PlateDetector(nn.Module):
    strides = (8,16,32)
    def __init__(self):
        super().__init__()
        channels = [1,32,64,128,256,256]
        self.stages = nn.ModuleList([nn.Sequential(block(a,b,2),block(b,b))
                                    for a,b in zip(channels[:-1],channels[1:])])
        self.lateral = nn.ModuleList([nn.Conv2d(c,128,1) for c in channels[3:]])
        self.smooth = nn.ModuleList([block(128,128) for _ in range(3)])
        self.cls_tower = nn.Sequential(block(128,128),block(128,128))
        self.box_tower = nn.Sequential(block(128,128),block(128,128))
        self.cls = nn.Conv2d(128,1,3,padding=1)
        self.box = nn.Conv2d(128,4,3,padding=1)
        self.center = nn.Conv2d(128,1,3,padding=1)
        nn.init.constant_(self.cls.bias, -math.log(99))

    def forward(self, x):
        features = []
        for i,stage in enumerate(self.stages):
            x = stage(x)
            if i >= 2:
                features.append(x)
        pyramid = [layer(f) for layer,f in zip(self.lateral,features)]
        for i in (1,0):
            pyramid[i] = pyramid[i] + F.interpolate(pyramid[i+1],size=pyramid[i].shape[-2:],mode="nearest")
        result = []
        for stride,smooth,p in zip(self.strides,self.smooth,pyramid):
            p = smooth(p)
            c, b = self.cls_tower(p), self.box_tower(p)
            result.append((self.cls(c), F.softplus(self.box(b))*stride, self.center(b)))
        return result

class CharacterClassifier(nn.Module):
    def __init__(self, classes: int = 36):
        super().__init__()
        self.features = nn.Sequential(block(1,32),nn.MaxPool2d(2),block(32,64),
                                      nn.MaxPool2d(2),block(64,128),nn.MaxPool2d(2))
        self.head = nn.Sequential(nn.Flatten(),nn.Linear(128*4*4,256),nn.ReLU(),
                                  nn.Dropout(0.25),nn.Linear(256,classes))
    def forward(self,x):
        return self.head(self.features(x))

def flatten_outputs(outputs):
    logits, distances, centers, points, strides = [],[],[],[],[]
    for stride,(cls,box,ctr) in zip(PlateDetector.strides,outputs):
        b,_,h,w = cls.shape
        yy,xx = torch.meshgrid(torch.arange(h,device=cls.device),torch.arange(w,device=cls.device),indexing="ij")
        points.append(torch.stack((xx.flatten()+0.5,yy.flatten()+0.5),1)*stride)
        strides.append(torch.full((h*w,),stride,device=cls.device))
        logits.append(cls.permute(0,2,3,1).reshape(b,-1))
        centers.append(ctr.permute(0,2,3,1).reshape(b,-1))
        distances.append(box.permute(0,2,3,1).reshape(b,-1,4))
    return (torch.cat(logits,1),torch.cat(distances,1),torch.cat(centers,1),
            torch.cat(points),torch.cat(strides))

def distances_to_boxes(points, distances):
    return torch.cat((points-distances[...,:2],points+distances[...,2:]),-1)
