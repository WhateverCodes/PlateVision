"""Small CPU trial using external labelled glyphs; never auto-promotes weights."""
import json,time
from pathlib import Path
from collections import Counter
import cv2,numpy as np,torch
from config import DEFAULTS,ALPHABET
from src.runtime import configure
from src.character_classifier import Classifier
from training.refine_verified_cnn import evaluate

def main():
 configure(2);torch.manual_seed(42);rng=np.random.default_rng(42)
 out=Path('models/external_glyph_trial_2026_10_03');out.mkdir(exist_ok=False)
 old=json.loads(Path('data/character_verified_300/verified_and_synthetic.json').read_text());new=json.loads(Path('data/dataset_trial_2026_10_03/characters.json').read_text());rows=old+new
 weights=[]
 for group in (old,new):
  counts=Counter(r['label'] for r in group)
  weights.extend(1/(len(counts)*counts[r['label']]) for r in group)
 weights=np.array(weights);weights/=weights.sum()
 images=np.stack([cv2.imread(r['image'],0) for r in rows]);labels=np.array([ALPHABET.index(r['label']) for r in rows])
 plates=json.loads(Path('data/accuracy_check_2026_09_21/val.json').read_text())+json.loads(Path('data/reader_reviewed_v2/val.json').read_text())
 reader=Classifier(DEFAULTS.adapted_character_path);baseline=evaluate(reader,plates);report={'baseline':baseline,'training_images':len(rows),'external_images':len(new),'note':'Same existing 45 development crops; external glyph parent/source independence unknown. No independent-test claim. Existing weights preserved.','trials':[]}
 print('Baseline', {k:v for k,v in baseline.items() if k!='results'},flush=True)
 start=time.monotonic()
 for lr in (1e-5,3e-5):
  torch.manual_seed(42);reader=Classifier(DEFAULTS.adapted_character_path);model=reader.model;opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
  for stage in range(1,4):
   losses=[];model.train()
   for module in model.modules():
    if isinstance(module,torch.nn.BatchNorm2d):module.eval()
   for step in range(60):
    if time.monotonic()-start>150:break
    ids=rng.choice(len(rows),32,p=weights);x=torch.from_numpy(images[ids]).float()[:,None]/255;y=torch.from_numpy(labels[ids]).long();opt.zero_grad();loss=torch.nn.functional.cross_entropy(model(x),y)
    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()));time.sleep(.03)
   if not losses:break
   model.eval();result=evaluate(reader,plates);result.update(lr=lr,stage=stage,steps=len(losses),loss=float(np.mean(losses)))
   dest=out/f'lr{lr}_stage{stage}.pt';torch.save(dict(kind='character_classifier',format_version=1,trained_steps=stage*60,alphabet=ALPHABET,model=model.state_dict(),provenance='Own CNN from-scratch lineage, adapted on 3500 externally labelled augmented glyphs with 1307 verified/synthetic replay. Source labels not individually verified; development trial.'),dest)
   result['checkpoint']=str(dest);report['trials'].append(result);print({k:v for k,v in result.items() if k!='results'},flush=True)
 report['seconds']=time.monotonic()-start;report['promoted']=False;(out/'report.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
