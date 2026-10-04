import json,shutil,hashlib
from pathlib import Path
p=Path('outputs/PlateVision1');source=p/'models/combined_paddle_2026_10_03';dest=p/'models/paddle_combined_v2';dest.mkdir(exist_ok=True)
for name in ['english.onnx','text_detector.onnx']:
    shutil.copy2(source/name,dest/name)
for name in ['PADDLEOCR_LICENSE.txt','RAPIDOCR_LICENSE.txt']:
    shutil.copy2(p/'models/paddle_plate_adapted_v1'/name,dest/name)
shutil.copy2(source/'report.json',dest/'training_report.json')
report=json.loads((source/'full_pipeline_evaluation.json').read_text())
rows=report['models']['candidate']['results']
sweep=[]
for threshold in [.90,.91,.92,.93,.94,.95]:
    sweep.append(dict(threshold=threshold,accepted_correct=sum(r['accepted'] and r['score']>=threshold and r['expected']==r['predicted'] for r in rows),accepted_wrong=sum(r['accepted'] and r['score']>=threshold and r['expected']!=r['predicted'] for r in rows)))
report['threshold_sweep']=sweep;report['selected_threshold']=.91
report['selection_note']='Development-selected: preserves 34 exact matches, reduces edits21 to18, increases accepted correct29 to31 at unchanged3 accepted wrong. Threshold tuning on these same45 crops is optimistic, not independent-test evidence.'
(dest/'development_evaluation.json').write_text(json.dumps(report,indent=2))
(dest/'adaptation.json').write_text(json.dumps(dict(training_lines=981,label_description='training lines with mixed human-reviewed and unverified source labels',method='Final CTC projection only, eight epochs, selected epoch1, 50% blend into prior adapted model; pretrained feature network unchanged',source_sha256=hashlib.sha256((p/'models/paddle_plate_adapted_v1/english.onnx').read_bytes()).hexdigest(),threshold=.91),indent=2))
print('Exported',dest)
