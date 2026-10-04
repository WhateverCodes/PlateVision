"""One bounded, class-balanced CNN experiment; preserves deployed weights."""
import json,time
from pathlib import Path
from collections import Counter
import cv2,numpy as np,torch
from config import DEFAULTS,ALPHABET
from src.runtime import configure
from src.character_classifier import Classifier
from training.refine_verified_cnn import evaluate

def run():
 configure(2);out=Path('outputs/evaluation/final_cnn_round')
 if out.exists():raise ValueError('Preserve existing experiment')
 out.mkdir(parents=True)
 rows=json.loads(Path('data/character_verified_300/verified_and_synthetic.json').read_text());counts=Counter(r['label'] for r in rows);verified=Counter(r['label'] for r in rows if r['origin']=='human_verified_glyph')
 images=np.stack([cv2.imread(r['image'],0) for r in rows]);assert images.shape[1:]==(32,32)
 labels=np.array([ALPHABET.index(r['label']) for r in rows]);weights=np.array([1/counts[r['label']] for r in rows]);weights/=weights.sum()
 valrows=json.loads(Path('data/synthetic/characters/val.json').read_text());vimages=np.stack([cv2.imread(str(Path('data/synthetic/characters')/r['image']),0) for r in valrows]);vlabels=torch.tensor([ALPHABET.index(r['label']) for r in valrows]);vx=torch.from_numpy(vimages).float()[:,None]/255
 plates=json.loads(Path('data/accuracy_check_2026_09_21/val.json').read_text())+json.loads(Path('data/reader_reviewed_v2/val.json').read_text())
 def glyph_metric(model):
  model.eval()
  with torch.no_grad():
   logits=torch.cat([model(batch) for batch in vx.split(32)])
   return {'accuracy':float((logits.argmax(1)==vlabels).float().mean()),'loss':float(torch.nn.functional.cross_entropy(logits,vlabels)),'images':len(valrows),'scope':'Synthetic validation glyphs, not real character accuracy'}
 base=Classifier(DEFAULTS.adapted_character_path);baseline={'plate':evaluate(base,plates),'synthetic_characters':glyph_metric(base.model)}
 report={'normalization':'32x32 white-on-black, divided by 255, identical training/inference','architecture':'Three Conv+BatchNorm+SiLU stages, two-layer classifier, dropout 0.25. Existing depth retained.','class_counts':{c:counts[c] for c in ALPHABET},'verified_class_counts':{c:verified[c] for c in ALPHABET},'baseline':baseline,'trials':[],'selection_rule':'Promote only >=18/45 exact, >=8 correct accepted and <=1 wrong accepted; otherwise freeze current model.','note':'Existing 45 development plate crops are repeatedly used for selection; not independent final-test performance. Training uses no plate evaluation records.'}
 started=time.monotonic()
 for rate in [3e-6,3e-5]:
  configure(2);classifier=Classifier(DEFAULTS.adapted_character_path);model=classifier.model;optimizer=torch.optim.AdamW(model.parameters(),lr=rate,weight_decay=1e-4);rng=np.random.default_rng(42);curve=[]
  for stage in range(2):
   model.train()
   for module in model.modules():
    if isinstance(module,torch.nn.BatchNorm2d):module.eval()
   losses=[]
   for step in range(40):
    if time.monotonic()-started>90:break
    indices=rng.choice(len(rows),24,p=weights);batch=[]
    for index in indices:
     matrix=cv2.getRotationMatrix2D((16,16),rng.uniform(-3,3),rng.uniform(.98,1.02));matrix[:,2]+=rng.uniform(-.7,.7,2);batch.append(cv2.warpAffine(images[index],matrix,(32,32),borderValue=0))
    x=torch.from_numpy(np.stack(batch)).float()[:,None]/255;y=torch.from_numpy(labels[indices]).long();optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(model(x),y)
    if not torch.isfinite(loss):raise ValueError('Invalid loss')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(float(loss.detach()));time.sleep(.03)
   metrics=evaluate(classifier,plates);curve.append({'stage':stage+1,'steps':len(losses),'train_loss':float(np.mean(losses)) if losses else None,'synthetic_validation':glyph_metric(model),'plate':metrics})
   print(rate,stage+1,{k:v for k,v in metrics.items() if k!='results'},flush=True)
  report['trials'].append({'lr':rate,'curve':curve})
 report['seconds']=time.monotonic()-started;report['promoted']=False
 (out/'report.json').write_text(json.dumps(report,indent=2));print('Baseline synthetic character score:',baseline['synthetic_characters'],flush=True)
if __name__=='__main__':run()
