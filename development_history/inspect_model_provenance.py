import sys
from pathlib import Path
sys.path.insert(0,str(Path('outputs/PlateVision8').resolve()))
from config import DEFAULTS
from src.plate_detector import load_checkpoint
for label,path,kind in [('plate detector',DEFAULTS.detector_path,'plate_detector'),('original CNN',DEFAULTS.character_path,'character_classifier'),('adapted CNN',DEFAULTS.adapted_character_path,'character_classifier')]:
 if path.exists():
  ck=load_checkpoint(path,kind);print(label,{k:v for k,v in ck.items() if k not in ['model','optimizer','rng_state'] and isinstance(v,(str,int,float,bool))})
print('Whole plate file exists:',DEFAULTS.reader_path.exists())
