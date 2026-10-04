"""Bounded CPU training with deterministic batch resume and separate reader checkpoints."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from torch import nn
from config import ALPHABET
from src.runtime import configure
from src.plate_reader import PlateReader,prepare_crop,decode
from src.preprocessing import read_image
from .data_tools import save_json

def batch(rows):
    images=torch.stack([prepare_crop(read_image(Path(r["image"]))) for r in rows])
    lengths=torch.tensor([len(r["text"]) for r in rows],dtype=torch.long)
    targets=torch.tensor([ALPHABET.index(c)+1 for r in rows for c in r["text"]],dtype=torch.long)
    return images,targets,lengths

def train(data,out,seconds=120,epochs=30,batch_size=8,max_steps=0):
    configure(2);data,out=Path(data),Path(out)
    if seconds<=0 or epochs<1 or batch_size<1 or max_steps<0:raise ValueError("Invalid training budget.")
    rows={s:json.loads((data/(s+".json")).read_text()) for s in ("train","val")}
    signature=hashlib.sha256((data/"train.json").read_bytes()+(data/"val.json").read_bytes()).hexdigest()
    for split in rows.values():
        for r in split:
            if hashlib.sha256(Path(r["image"]).read_bytes()).hexdigest()!=r["file_sha256"]:raise ValueError("Image changed since dataset snapshot.")
    existing=torch.load(out/"last.pt",map_location="cpu",weights_only=True) if (out/"last.pt").exists() else None
    architecture=existing.get("architecture",1) if existing else 2
    model=PlateReader(architecture);optimizer=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.0001)
    criterion=nn.CTCLoss(blank=0,zero_infinity=False)
    state={"epoch":0,"cursor":0,"steps":0,"best":float("inf"),"history":[],"partial_loss":0.,"partial_count":0}
    out.mkdir(parents=True,exist_ok=True);last=out/"last.pt"
    if last.exists():
        checkpoint=torch.load(last,map_location="cpu",weights_only=True)
        if checkpoint.get("kind")!="plate_reader" or checkpoint.get("format_version")!=1 or checkpoint["dataset_fingerprint"]!=signature or checkpoint["batch_size"]!=batch_size:raise ValueError("Checkpoint does not match dataset or batch size.")
        model.load_state_dict(checkpoint["model"]);optimizer.load_state_dict(checkpoint["optimizer"]);state=checkpoint["state"]
    started=time.monotonic();initial_steps=state["steps"]
    def expired():return time.monotonic()-started>=seconds or bool(max_steps and state["steps"]-initial_steps>=max_steps)
    def save(name,status):
        temp=out/(name+".tmp")
        torch.save({"kind":"plate_reader","format_version":1,"architecture":architecture,"alphabet":ALPHABET,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"state":state,"dataset_fingerprint":signature,"batch_size":batch_size,"status":status,"trained_steps":state["steps"],"provenance":"Random initialization; "+str(data.resolve())},temp);temp.replace(out/name)
        save_json(out/"history.json",state["history"])
    def loss_for(items):
        x,y,lengths=batch(items);pred=model(x)
        loss=criterion(pred,y,torch.full((len(items),),pred.shape[0],dtype=torch.long),lengths)
        if not torch.isfinite(loss):raise ValueError("Non-finite CTC loss. Inspect data.")
        return loss,pred
    status="completed"
    try:
        while state["epoch"]<epochs:
            order=torch.randperm(len(rows["train"]),generator=torch.Generator().manual_seed(42+state["epoch"])).tolist();model.train()
            while state["cursor"]<len(order):
                if expired():status="paused-budget";break
                chosen=order[state["cursor"]:state["cursor"]+batch_size];items=[rows["train"][i] for i in chosen]
                optimizer.zero_grad();loss,_=loss_for(items);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                state["cursor"]+=len(items);state["steps"]+=1;state["partial_loss"]+=float(loss.detach())*len(items);state["partial_count"]+=len(items)
                if state["steps"]%10==0:print(f'Epoch {state["epoch"]+1} | {state["cursor"]}/{len(order)} | loss {float(loss.detach()):.4f}',flush=True)
                time.sleep(0.05)
            if expired():status="paused-budget";break
            model.eval();val_loss=0.;correct=0
            with torch.inference_mode():
                for start in range(0,len(rows["val"]),batch_size):
                    if expired():status="paused-validation";break
                    items=rows["val"][start:start+batch_size];loss,pred=loss_for(items);val_loss+=float(loss)*len(items);correct+=sum(a==b["text"] for a,b in zip(decode(pred),items))
            if status=="paused-validation":break
            val_loss/=len(rows["val"])
            entry={"epoch":state["epoch"]+1,"train_loss":state["partial_loss"]/state["partial_count"],"val_loss":val_loss,"val_exact_accuracy":correct/len(rows["val"])}
            state["history"].append(entry);state["epoch"]+=1;state["cursor"]=0;state["partial_loss"]=0.;state["partial_count"]=0
            improved=val_loss<state["best"];state["best"]=min(state["best"],val_loss)
            print(json.dumps(entry),flush=True);save("last.pt","epoch-complete")
            if improved:save("best.pt","best-validation-loss")
    except KeyboardInterrupt:status="paused-user"
    except Exception:
        status="failed"
        raise
    finally:
        save("last.pt",status)
    elapsed=time.monotonic()-started
    report={"status":status,"new_steps":state["steps"]-initial_steps,"training_seconds":elapsed,"seconds_per_step":elapsed/max(1,state["steps"]-initial_steps),"epoch":state["epoch"],"cursor":state["cursor"],"parameters":sum(p.numel() for p in model.parameters()),"threads":2,"batch_size":batch_size,"checkpoint":str(last.resolve()),"note":"Timing excludes startup and file integrity checks. This is not evidence of recognition accuracy."}
    save_json(out/"last_session.json",report);print(json.dumps(report,indent=2));return report
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True);p.add_argument("--out",type=Path,required=True);p.add_argument("--seconds",type=float,default=120);p.add_argument("--epochs",type=int,default=30);p.add_argument("--batch-size",type=int,default=8);p.add_argument("--max-steps",type=int,default=0);a=p.parse_args();train(a.data,a.out,a.seconds,a.epochs,a.batch_size,a.max_steps)
