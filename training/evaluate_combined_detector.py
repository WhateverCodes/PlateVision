"""One-class Indian development benchmark for own and retained external detectors."""
import argparse,json
from pathlib import Path
from src.runtime import configure
from src.plate_detector import Detector
from src.pretrained import PretrainedDetector
from src.preprocessing import read_image
from config import DEFAULTS
from training.evaluate_plate_detector import metrics

def main():
 a=argparse.ArgumentParser();a.add_argument('--weights');a.add_argument('--out',required=True);args=a.parse_args();configure(2)
 rows=json.loads(Path('data/combined_training_2026_10_03/detector/val.json').read_text());readers={'own':Detector(Path(args.weights) if args.weights else DEFAULTS.detector_path,threshold=.01)}
 if not args.weights:readers['deployed_yolo']=PretrainedDetector(DEFAULTS.pretrained_folder,.01,2)
 report={'scope':'Indian grouped development validation, source boxes not fully reviewed. Not independent final-test accuracy; detector candidates limited to30.','images':len(rows),'models':{}}
 for name,detector in readers.items():
  pred=[];truth=[]
  for r in rows:pred.append(detector(read_image(Path(r['image']))));truth.append(r['boxes'])
  m=metrics(pred,truth,.5,.5);report['models'][name]={k:v for k,v in m.items() if not k.startswith('curve')};print(name,report['models'][name],flush=True)
 Path(args.out).parent.mkdir(parents=True,exist_ok=True);Path(args.out).write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
