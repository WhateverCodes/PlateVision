"""Held-out character metrics and a labeled 36-class confusion matrix."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from config import DEFAULTS,ROOT
from src.runtime import configure
from src.character_classifier import Classifier
from .dataset import VisionDataset
from .data_tools import save_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--weights",type=Path,default=DEFAULTS.character_path)
    p.add_argument("--out",type=Path,default=ROOT/"outputs/evaluation/characters")
    a=p.parse_args()
    configure()
    classifier=Classifier(a.weights)
    data=VisionDataset(a.data/"test.json","character_classifier",alphabet=classifier.alphabet)
    actual,predicted=[],[]
    with torch.inference_mode():
        for x,y in DataLoader(data,batch_size=32,num_workers=0):
            actual.extend(y.tolist())
            predicted.extend(classifier.model(x).argmax(1).tolist())
    labels=list(range(len(classifier.alphabet)))
    report=classification_report(actual,predicted,labels=labels,target_names=list(classifier.alphabet),output_dict=True,zero_division=0)
    report["accuracy"]=accuracy_score(actual,predicted)
    report["provenance"]=classifier.provenance
    matrix=confusion_matrix(actual,predicted,labels=labels)
    save_json(a.out/"metrics.json",report)
    save_json(a.out/"confusion_matrix.json",matrix.tolist())
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig,ax=plt.subplots(figsize=(12,10))
    im=ax.imshow(matrix,cmap="Blues")
    ax.set_xticks(labels,list(classifier.alphabet)); ax.set_yticks(labels,list(classifier.alphabet))
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.colorbar(im,ax=ax); fig.tight_layout()
    fig.savefig(a.out/"confusion_matrix.png"); plt.close(fig)
    print(json.dumps({"accuracy":report["accuracy"],"macro avg":report["macro avg"]},indent=2))

if __name__=="__main__":
    main()
