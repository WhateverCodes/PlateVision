"""Separate pretrained general COCO detector; never used to read a plate."""
from pathlib import Path
import cv2
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

VEHICLE_CLASSES = {3:"Car",4:"Motorcycle",6:"Bus",8:"Truck"}

class VehicleDetector:
    def __init__(self,path: Path, threshold: float = 0.5):
        if not path.is_file():
            raise ValueError("Vehicle weights missing. Run: python -m training.download_vehicle_weights")
        self.model = ssdlite320_mobilenet_v3_large(weights=None,weights_backbone=None,num_classes=91).eval()
        self.model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
        self.threshold = threshold

    @torch.inference_mode()
    def __call__(self,image):
        rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255
        output = self.model([tensor])[0]
        return [{"box":box.int().tolist(),"vehicle_type":VEHICLE_CLASSES[int(label)],"score":float(score)}
                for box,label,score in zip(output["boxes"],output["labels"],output["scores"])
                if int(label) in VEHICLE_CLASSES and float(score)>=self.threshold]
