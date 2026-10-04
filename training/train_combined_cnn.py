"""Sequential all-usable-glyph pass with verified replay, bounded CPU trial."""
import json,time
from pathlib import Path
from collections import Counter
import cv2,numpy as np,torch
from config import DEFAULTS,ALPHABET
from src.runtime import configure
from src.character_classifier import Classifier
from training.refine_verified_cnn import evaluate

def main():
 configure(2);out=Path('models/combined_cnn_2026_10_03');out.mkdir(exist_ok=False)
 rows=json.loads(Path('data/combined_training_2026_10_03/characters/train_with_real_plates.json').read_text());x=np.stack([cv2.imread(r['image'],0) for r in rows]);y=np.array([ALPHABET.index(r['label']) for r in rows]);origin=[r['origin'] for r in rows];old=np.array([i for i,o in enumerate(origin) if o in ('human_verified_glyph','synthetic')]);new=np.array([i for i,o in enumerate(origin) if o not in ('human_verified_glyph','synthetic')]);counts=Counter(r['label'] for r in rows)
 weights=np.array([min(4.,len(rows)/(36*counts[r['label']]))*(3. if r['origin']=='human_verified_glyph' else .15 if r['origin']=='EMNIST_byclass_training' else .5 if r['origin'] in ('archive7_augmented','combined_weak_alignment') else 1.) for r in rows],np.float32)
 plates=json.loads(Path('data/accuracy_check_2026_09_21/val.json').read_text())+json.loads(Path('data/reader_reviewed_v2/val.json').read_text());reader=Classifier(DEFAULTS.adapted_character_path);base=evaluate(reader,plates);model=reader.model;optimizer=torch.optim.AdamW(model.parameters(),lr=5e-6,weight_decay=1e-4);rng=np.random.default_rng(42);start=time.monotonic();steps=0;seen=set();history=[]
 print('Baseline',{k:v for k,v in base.items() if k!='results'},'training rows',len(rows),flush=True)
 for epoch in range(1,3):
  order=rng.permutation(new);losses=[];model.train()
  for m in model.modules():
   if isinstance(m,torch.nn.BatchNorm2d):m.eval()
  for offset in range(0,len(order),64):
   if time.monotonic()-start>480:break
   portion=order[offset:offset+64];ids=np.concatenate([portion,rng.choice(old,16)]);seen.update(map(int,portion));inputs=torch.from_numpy(x[ids]).float()[:,None]/255;labels=torch.from_numpy(y[ids]).long();w=torch.from_numpy(weights[ids]);optimizer.zero_grad();losses_raw=torch.nn.functional.cross_entropy(model(inputs),labels,reduction='none');loss=(losses_raw*w).sum()/w.sum()
   if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();steps+=1;losses.append(float(loss.detach()));time.sleep(.02)
   if steps%100==0:print('epoch',epoch,'steps',steps,'unique new examples',len(seen),'loss',round(losses[-1],4),flush=True)
  model.eval();metric=evaluate(reader,plates);checkpoint=out/f'epoch{epoch}.pt';torch.save(dict(kind='character_classifier',format_version=1,trained_steps=steps,alphabet=ALPHABET,model=model.state_dict(),provenance='Own from-scratch CNN lineage; combined external printed glyphs, EMNIST training sample, training-only weak real plate glyphs and verified/synthetic replay. Development evaluated.'),checkpoint);history.append(dict(epoch=epoch,loss=float(np.mean(losses)),checkpoint=str(checkpoint),**metric));print('Evaluation',{k:v for k,v in history[-1].items() if k!='results'},flush=True)
  if time.monotonic()-start>480:break
 (out/'report.json').write_text(json.dumps(dict(baseline=base,history=history,training_images=len(rows),new_examples=len(new),unique_new_examples_seen=len(seen),seconds=time.monotonic()-start,promoted=False,note='Development crop scores, not independent-test accuracy. Weak source labels not individually verified.'),indent=2))
if __name__=='__main__':main()
