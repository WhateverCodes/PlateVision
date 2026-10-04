"""Own CNN character recognition. Input normalization is shared with training."""
from pathlib import Path
import numpy as np
import torch
from .architectures import CharacterClassifier
from .plate_detector import load_checkpoint

class Classifier:
    def __init__(self,path: Path):
        ckpt = load_checkpoint(path,"character_classifier")
        self.alphabet = ckpt["alphabet"]
        self.model = CharacterClassifier(len(self.alphabet)).eval()
        self.model.load_state_dict(ckpt["model"])
        self.provenance = ckpt.get("provenance","Not recorded")

    @torch.inference_mode()
    def __call__(self, chars) -> tuple[str,list[float]]:
        if not chars:
            return "",[]
        x = torch.from_numpy(np.stack([c.image for c in chars])).float()[:,None]/255
        probs = self.model(x).softmax(1)
        confidence,indices = probs.max(1)
        return "".join(self.alphabet[i] for i in indices.tolist()),confidence.tolist()


    def read_segmented(self,plate,threshold=0.75):
        from .character_segmenter import segmentation_candidates
        from .validator import plausible
        candidates=segmentation_candidates(plate)
        baseline=max(range(len(candidates)),key=lambda i:candidates[i][0])
        choices=[]
        for _,chars,mask in candidates:
            text,scores=self(chars)
            quality=float(np.mean(np.log(np.clip(scores,1e-8,1)))) if scores else -100.
            choices.append((quality,chars,mask,text,scores))
        eligible=[v for v in choices if plausible(v[3]) and v[4] and min(v[4])>=threshold]
        readable=[v for v in choices if plausible(v[3]) and v[4]]
        selected=max(eligible or readable,key=lambda v:v[0]) if eligible or readable else choices[baseline]
        _,chars,mask,text,scores=selected
        return chars,mask,text,scores
