"""Time-bounded CPU-first training, resumable at a batch boundary."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset
from config import ROOT,ALPHABET
from src.runtime import configure
from src.architectures import PlateDetector,CharacterClassifier
from .dataset import VisionDataset,collate
from .losses import detection_loss
from .data_tools import save_json

def arguments(kind):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--out",type=Path,default=ROOT/"models"/kind)
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--batch-size",type=int,default=2 if kind=="plate_detector" else 32)
    p.add_argument("--image-size",type=int,default=320)
    p.add_argument("--lr",type=float,default=0.001)
    p.add_argument("--threads",type=int,choices=[1,2,3,4],default=2)
    p.add_argument("--max-minutes",type=float,default=10)
    p.add_argument("--cooldown",type=float,default=0.05,help="Seconds to rest after each training batch")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    p.add_argument("--resume",type=Path)
    p.add_argument("--initialize",type=Path,help="Initialize model weights only from an earlier own-model checkpoint; resets optimizer and epoch.")
    p.add_argument("--no-augmentation",action="store_true")
    p.add_argument("--limit-train",type=int,default=0,help="Smoke tests only; recorded in checkpoint")
    p.add_argument("--limit-val",type=int,default=0,help="Smoke tests only; recorded in checkpoint")
    return p

def train(kind):
    p=arguments(kind)
    a=p.parse_args()
    if a.resume and a.initialize:
        p.error("Use either resume or initialize, not both.")
    if a.epochs<1 or a.batch_size<1 or a.max_minutes<=0 or a.cooldown<0:
        p.error("Epochs, batch and time budget must be positive; cooldown must be nonnegative.")
    if a.image_size<64 or a.image_size%32:
        p.error("Image size must be >=64 and divisible by 32.")
    configure(a.threads,a.seed)
    if a.device=="cuda" and not torch.cuda.is_available():
        p.error("CUDA is unavailable. Use --device cpu on your Intel laptop.")
    device=torch.device(a.device)
    alphabet_path=a.data/"alphabet.json"
    alphabet=json.loads(alphabet_path.read_text()) if alphabet_path.exists() else ALPHABET
    trainset=VisionDataset(a.data/"train.json",kind,a.image_size,alphabet,not a.no_augmentation)
    valset=VisionDataset(a.data/"val.json",kind,a.image_size,alphabet,False)
    if a.limit_train:
        trainset.rows=trainset.rows[:a.limit_train]
    if a.limit_val:
        valset.rows=valset.rows[:a.limit_val]
    model=(PlateDetector() if kind=="plate_detector" else CharacterClassifier(len(alphabet))).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    fingerprint=hashlib.sha256((a.data/"train.json").read_bytes()+(a.data/"val.json").read_bytes()).hexdigest()
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    epoch,cursor,steps,best=0,0,0,float("inf")
    history=[]
    partial_sum,partial_count=0.,0
    if a.resume:
        ckpt=torch.load(a.resume,map_location=device,weights_only=True)
        if ckpt["kind"]!=kind or ckpt["dataset_fingerprint"]!=fingerprint or ckpt["alphabet"]!=alphabet:
            raise ValueError("Resume checkpoint does not match model, dataset or alphabet.")
        for key in ("batch_size","image_size","seed","no_augmentation","limit_train","limit_val"):
            if ckpt["config"][key]!=config[key]:
                raise ValueError(f"Keep --{key.replace('_','-')} unchanged when resuming.")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        epoch,cursor,steps,best=ckpt["epoch"],ckpt["cursor"],ckpt["trained_steps"],ckpt["best"]
        history=ckpt["history"]
        partial_sum,partial_count=ckpt["partial_sum"],ckpt["partial_count"]
        torch.set_rng_state(ckpt["torch_rng"].cpu())
        random.setstate(ckpt["python_rng"])
        state=ckpt["numpy_rng"]
        np.random.set_state((state[0],np.array(state[1],dtype=np.uint32),state[2],state[3],state[4]))
    elif (a.out/"last.pt").exists():
        raise ValueError("Checkpoint already exists. Use --resume or a new --out directory.")
    if a.initialize:
        initial=torch.load(a.initialize,map_location=device,weights_only=True)
        if initial.get("kind")!=kind or initial.get("alphabet")!=alphabet:
            raise ValueError("Initialization model kind or alphabet mismatch.")
        model.load_state_dict(initial["model"])
    a.out.mkdir(parents=True,exist_ok=True)
    card_path=a.data/"dataset_card.json"
    provenance=json.loads(card_path.read_text()).get("provenance","Unspecified") if card_path.exists() else "Unspecified"
    if a.resume:provenance=ckpt.get("provenance",provenance)
    if a.initialize:provenance += " | Initialized from own checkpoint: " + str(a.initialize)
    start=time.monotonic()
    def timed_out():
        return time.monotonic()-start>=a.max_minutes*60
    def save(name,status):
        rng=np.random.get_state()
        ckpt={"format_version":1,"kind":kind,"model":model.state_dict(),"optimizer":optimizer.state_dict(),
              "image_size":a.image_size,"alphabet":alphabet,"epoch":epoch,"cursor":cursor,
              "trained_steps":steps,"best":best,"history":history,"config":config,
              "dataset_fingerprint":fingerprint,"provenance":provenance,"status":status,
              "partial_sum":partial_sum,"partial_count":partial_count,
              "torch_rng":torch.get_rng_state(),"python_rng":random.getstate(),
              "numpy_rng":(rng[0],rng[1].tolist(),rng[2],rng[3],rng[4])}
        tmp=a.out/(name+".tmp")
        torch.save(ckpt,tmp)
        tmp.replace(a.out/name)
        save_json(a.out/"history.json",history)
        save_json(a.out/"training_config.json",config)
    def loss_for(x,y):
        if kind=="plate_detector":
            return detection_loss(model(x),[t.to(device) for t in y])
        return nn.functional.cross_entropy(model(x),torch.stack(y).to(device))
    try:
        while epoch<a.epochs:
            # Dedicated generators avoid changing model/dropout RNG on resume.
            order=torch.randperm(len(trainset),generator=torch.Generator().manual_seed(a.seed+epoch)).tolist()
            loader=DataLoader(Subset(trainset,order[cursor:]),batch_size=a.batch_size,collate_fn=collate,
                              num_workers=0,generator=torch.Generator().manual_seed(a.seed+epoch))
            model.train()
            for x,y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss=loss_for(x.to(device),y)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss; inspect dataset and learning rate.")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),10)
                optimizer.step()
                n=len(y)
                partial_sum+=float(loss.detach())*n
                partial_count+=n
                cursor+=n
                steps+=1
                if steps%10==0:
                    print(f"Epoch {epoch+1} | {cursor}/{len(trainset)} | loss {float(loss.detach()):.4f}",flush=True)
                time.sleep(a.cooldown)
                if timed_out():
                    save("last.pt","paused-time-budget")
                    print("Time budget reached. Saved last.pt; resume to continue.",flush=True)
                    return
            model.eval()
            val_sum,val_count=0.,0
            loader=DataLoader(valset,batch_size=a.batch_size,collate_fn=collate,num_workers=0,
                              generator=torch.Generator().manual_seed(a.seed))
            with torch.inference_mode():
                for x,y in loader:
                    if timed_out():
                        save("last.pt","paused-validation")
                        print("Time budget reached during validation. Resume to finish validation.",flush=True)
                        return
                    value=loss_for(x.to(device),y)
                    val_sum+=float(value)*len(y)
                    val_count+=len(y)
            val_loss=val_sum/max(val_count,1)
            history.append({"epoch":epoch+1,"train_loss":partial_sum/max(partial_count,1),"val_loss":val_loss})
            print(json.dumps(history[-1]),flush=True)
            improved=val_loss<best
            best=min(best,val_loss)
            epoch+=1
            cursor=0
            partial_sum,partial_count=0.,0
            save("last.pt","epoch-complete")
            if improved:
                save("best.pt","best-validation-loss")
            if len(history)>=6 and min(h["val_loss"] for h in history[-5:])>best:
                print("Early stopping: no improvement in five completed epochs.",flush=True)
                break
    except KeyboardInterrupt:
        # Ctrl+C can occur inside an optimizer step. Last fully saved epoch/budget checkpoint is safest.
        save("interrupted.pt","interrupted-may-repeat-last-batch")
        print("Interrupted. Saved interrupted.pt. Prefer last.pt when available.",flush=True)
    finally:
        if history:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
            for key in ("train_loss","val_loss"):
                plt.plot([h["epoch"] for h in history],[h[key] for h in history],label=key)
            plt.xlabel("Completed epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(a.out/"loss.png")
            plt.close()
