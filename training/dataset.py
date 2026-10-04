"""Disk-streamed datasets: workers=0, no eager image caching."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from src.preprocessing import read_image,grayscale,letterbox
from .augmentations import augment

class VisionDataset(Dataset):
    def __init__(self,manifest: Path,kind: str,image_size: int=320,alphabet: str="",augmentation: bool=False):
        self.manifest=manifest
        self.rows=json.loads(manifest.read_text(encoding="utf-8"))
        if not self.rows:
            raise ValueError(f"Empty dataset: {manifest}")
        self.kind,self.image_size,self.alphabet,self.augmentation=kind,image_size,alphabet,augmentation
        if kind=="character_classifier" and any(r.get("label") not in alphabet for r in self.rows):
            raise ValueError("Character labels do not match the configured alphabet.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index]
        gray=grayscale(read_image(self.manifest.parent/row["image"]))
        if self.kind=="plate_detector":
            image,scale,offset=letterbox(gray,self.image_size)
            boxes=np.array(row["boxes"],dtype=np.float32).reshape(-1,4)
            boxes=boxes*scale+np.array([*offset,*offset],np.float32)
            if self.augmentation:
                image,boxes=augment(image,boxes)
            target=torch.from_numpy(boxes.astype(np.float32))
        else:
            if gray.shape!=(32,32):
                raise ValueError("Character images must be normalized to 32x32 white-on-black.")
            image=gray
            if self.augmentation:
                image,_=augment(image)
            target=torch.tensor(self.alphabet.index(row["label"]),dtype=torch.long)
        return torch.from_numpy(image.copy()).float()[None]/255,target

def collate(batch):
    images,targets=zip(*batch)
    return torch.stack(images),list(targets)
