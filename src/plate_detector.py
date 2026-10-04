"""Checkpoint-backed inference; missing or incompatible weights fail explicitly."""
from pathlib import Path
import numpy as np
import torch
from torchvision.ops import nms
from .architectures import PlateDetector, flatten_outputs, distances_to_boxes
from .preprocessing import grayscale, letterbox

def load_checkpoint(path: Path, kind: str) -> dict:
    if not path.is_file():
        raise ValueError(f"Missing {kind} model. Follow the training steps in README.md.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("kind") != kind or checkpoint.get("format_version") != 1:
        raise ValueError(f"Incompatible {kind} checkpoint.")
    if not checkpoint.get("trained_steps",0):
        raise ValueError(f"The {kind} checkpoint has not been trained.")
    return checkpoint

class Detector:
    def __init__(self, path: Path, threshold: float = 0.5, nms_iou: float = 0.4):
        ckpt = load_checkpoint(path,"plate_detector")
        self.model = PlateDetector().eval()
        self.model.load_state_dict(ckpt["model"])
        self.size = int(ckpt["image_size"])
        self.threshold, self.nms_iou = threshold,nms_iou
        self.provenance = ckpt.get("provenance","Not recorded")

    @torch.inference_mode()
    def __call__(self, image: np.ndarray) -> list[dict]:
        gray, scale, (ox,oy) = letterbox(grayscale(image), self.size)
        tensor = torch.from_numpy(gray.copy()).float()[None,None]/255
        return self.decode(self.model(tensor), scale,(ox,oy),image.shape[:2])

    def decode(self, outputs, scale, offset, shape):
        cls,dist,ctr,points,_ = flatten_outputs(outputs)
        scores = (cls[0].sigmoid()*ctr[0].sigmoid()).sqrt()
        selected = torch.where(scores >= self.threshold)[0]
        selected = selected[scores[selected].argsort(descending=True)[:1000]]
        boxes = distances_to_boxes(points[selected],dist[0,selected])
        keep = nms(boxes,scores[selected],self.nms_iou)[:30]
        results = []
        h,w = shape
        for idx in keep:
            box = boxes[idx].cpu().numpy()
            box[[0,2]] = (box[[0,2]]-offset[0])/scale
            box[[1,3]] = (box[[1,3]]-offset[1])/scale
            box[[0,2]] = np.clip(box[[0,2]],0,w)
            box[[1,3]] = np.clip(box[[1,3]],0,h)
            if box[2]-box[0] >= 10 and box[3]-box[1] >= 6:
                results.append({"box":box.astype(int).tolist(),"score":float(scores[selected[idx]])})
        return results
