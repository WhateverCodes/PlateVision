"""Bounded own-CNN adaptation with verified-label weighting; development selection only."""
import json,time,argparse
from pathlib import Path
from dataclasses import replace
import cv2,numpy as np,torch
from config import DEFAULTS,ALPHABET
from src.runtime import configure
from src.character_classifier import Classifier
from src.pipeline import Pipeline
from src.classical_detector import PlateCropDetector
from src.preprocessing import read_image
from training.evaluate_plate_reader import edit_distance

def evaluate(model,records):
    model.model.eval();pipe=Pipeline(replace(DEFAULTS,reader_mode='characters',localization_mode='crop',rectify=False),PlateCropDetector(),model)
    rows=[]
    for r in records:
        d=pipe(read_image(Path(r['image'])))[0]
        rows.append(dict(id=r['id'],expected=r['text'],predicted=d['raw_text'],accepted=d['recognized']))
    return dict(exact=sum(r['expected']==r['predicted'] for r in rows),accepted_correct=sum(r['accepted'] and r['expected']==r['predicted'] for r in rows),accepted_wrong=sum(r['accepted'] and r['expected']!=r['predicted'] for r in rows),character_errors=sum(edit_distance(r['expected'],r['predicted']) for r in rows),results=rows)

def main():
    configure(2);p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--base',type=Path,default=Path('models/character_verified_100/last.pt'));p.add_argument('--data',type=Path,default=Path('data/character_verified_100/train.json'));p.add_argument('--freeze-features',action='store_true');a=p.parse_args()
    if a.out.exists():raise ValueError('Use a fresh experiment directory')
    a.out.mkdir(parents=True)
    records=json.loads(Path('data/accuracy_check_2026_09_21/val.json').read_text())+json.loads(Path('data/reader_reviewed_v2/val.json').read_text())
    rows=json.loads(a.data.read_text())
    model=Classifier(a.base);baseline=evaluate(model,records)
    x=np.stack([cv2.imread(r['image'],0) for r in rows]);y=np.array([ALPHABET.index(r['label']) for r in rows])
    weights=np.array([12 if r['origin']=='human_verified_glyph' else (.25 if r['origin']=='weak_alignment' else 1) for r in rows],float);weights/=weights.sum()
    if a.freeze_features:
        for parameter in model.model.features.parameters():parameter.requires_grad_(False)
    opt=torch.optim.AdamW([v for v in model.model.parameters() if v.requires_grad],lr=1e-5,weight_decay=1e-4)
    rng=np.random.default_rng(42);start=time.monotonic();reports={'baseline':baseline,'candidates':[],'note':'45 development crops repeatedly used for selection; not an independent final evaluation. No evaluation examples used in gradient updates.'}
    for stage in range(3):
        model.model.train();losses=[]
        if a.freeze_features:model.model.features.eval()
        for step in range(60):
            if time.monotonic()-start>90:break
            ids=rng.choice(len(rows),24,p=weights);images=[]
            for i in ids:
                im=x[i];h,w=im.shape
                m=cv2.getRotationMatrix2D((w/2,h/2),rng.uniform(-4,4),rng.uniform(.96,1.04));m[:,2]+=rng.uniform(-1,1,2)
                images.append(cv2.warpAffine(im,m,(w,h),borderMode=cv2.BORDER_CONSTANT,borderValue=0))
            inputs=torch.from_numpy(np.stack(images)).float()[:,None]/255;labels=torch.from_numpy(y[ids]).long()
            opt.zero_grad();loss=torch.nn.functional.cross_entropy(model.model(inputs),labels)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.model.parameters(),5);opt.step();losses.append(float(loss.detach()));time.sleep(.03)
        if not losses:break
        result=evaluate(model,records);result.update(stage=stage+1,steps=len(losses),mean_loss=float(np.mean(losses)))
        path=a.out/f'stage_{stage+1}.pt'
        torch.save(dict(kind='character_classifier',format_version=1,trained_steps=sum(v['steps'] for v in reports['candidates'])+len(losses),alphabet=ALPHABET,model=model.model.state_dict(),provenance='Own CNN: verified-glyph weighted adaptation with gentle geometric augmentation, synthetic replay and downweighted weak labels. Development-selected; not pretrained OCR.'),path)
        reports['candidates'].append(result);print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)
    reports['elapsed_seconds']=time.monotonic()-start
    (a.out/'evaluation.json').write_text(json.dumps(reports,indent=2))
    print('Baseline', {k:v for k,v in baseline.items() if k!='results'},flush=True)
if __name__=='__main__':main()

