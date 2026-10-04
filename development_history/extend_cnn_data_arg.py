from pathlib import Path
p=Path('outputs/PlateVision1/training/refine_verified_cnn.py');s=p.read_text(encoding='utf8');s=s.replace(";a=p.parse_args()",";p.add_argument('--data',type=Path,default=Path('data/character_verified_100/train.json'));a=p.parse_args()")
s=s.replace("rows=json.loads(Path('data/character_verified_100/train.json').read_text())","rows=json.loads(a.data.read_text())")
p.write_text(s,encoding='utf8')
