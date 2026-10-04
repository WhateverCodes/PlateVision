from pathlib import Path
p=Path('outputs/PlateVision1/training/refine_verified_cnn.py');s=p.read_text(encoding='utf-8-sig');s=s.replace("p.add_argument('--out',type=Path,required=True);a=p.parse_args()", "p.add_argument('--out',type=Path,required=True);p.add_argument('--base',type=Path,default=Path('models/character_verified_100/last.pt'));a=p.parse_args()")
s=s.replace('model=Classifier(DEFAULTS.adapted_character_path);baseline=evaluate(model,records)','model=Classifier(a.base);baseline=evaluate(model,records)')
s=s.replace('trained_steps=(stage+1)*60','trained_steps=sum(v[\'steps\'] for v in reports[\'candidates\'])+len(losses)')
p.write_text(s,encoding='utf8')
