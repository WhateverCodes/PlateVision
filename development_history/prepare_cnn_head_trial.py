from pathlib import Path
import json
p=Path('outputs/PlateVision1/training/refine_verified_cnn.py');s=p.read_text();s=s.replace(";a=p.parse_args()",";p.add_argument('--freeze-features',action='store_true');a=p.parse_args()")
s=s.replace("opt=torch.optim.AdamW(model.model.parameters(),lr=1e-5,weight_decay=1e-4)","if a.freeze_features:\n        for parameter in model.model.features.parameters():parameter.requires_grad_(False)\n    opt=torch.optim.AdamW([v for v in model.model.parameters() if v.requires_grad],lr=1e-5,weight_decay=1e-4)")
s=s.replace('model.model.train();losses=[]','model.model.train();losses=[]\n        if a.freeze_features:model.model.features.eval()')
p.write_text(s)
p=Path('outputs/PlateVision1/data/character_verified_300');rows=json.loads((p/'train.json').read_text());clean=[r for r in rows if r['origin']!='weak_alignment'];(p/'verified_and_synthetic.json').write_text(json.dumps(clean,indent=2));print('Clean head-training examples:',len(clean))
