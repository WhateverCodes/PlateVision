import json
from pathlib import Path
from src.runtime import configure
from src.character_classifier import Classifier
from src.character_segmenter import segment
from src.preprocessing import read_image
from src.validator import plausible
from training.evaluate_plate_reader import edit_distance
from training.data_tools import save_json

def main(model_path="models/character_classifier/best.pt",output="outputs/evaluation/character_candidates_validation.json", data="data/reader_v1/val.json"):
    configure(2);c=Classifier(Path(model_path));results=[]
    for r in json.loads(Path(data).read_text()):
        image=read_image(Path(r["image"]));chars,_=segment(image);baseline,_=c(chars)
        _,_,text,scores=c.read_segmented(image)
        results.append({"expected":r["text"],"baseline":baseline,"predicted":text,"edit_distance":edit_distance(text,r["text"]),"accepted":plausible(text) and bool(scores) and min(scores)>=.75})
    report={"model":str(model_path),"images":len(results),"baseline_exact_matches":sum(r["expected"]==r["baseline"] for r in results),"exact_matches":sum(r["expected"]==r["predicted"] for r in results),"accepted_correct":sum(r["expected"]==r["predicted"] and r["accepted"] for r in results),"accepted_wrong":sum(r["expected"]!=r["predicted"] and r["accepted"] for r in results),"character_error_rate":sum(r["edit_distance"] for r in results)/sum(len(r["expected"]) for r in results),"results":results,"note":"Small difficult development validation set used during method selection. Not independent final-test accuracy."}
    save_json(Path(output),report);print(json.dumps({k:v for k,v in report.items() if k!="results"},indent=2))
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--model",default="models/character_classifier/best.pt");p.add_argument("--out",default="outputs/evaluation/character_candidates_validation.json");p.add_argument("--data",default="data/reader_v1/val.json");a=p.parse_args();main(a.model,a.out,a.data)
