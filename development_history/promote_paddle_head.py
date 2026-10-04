from pathlib import Path
import json,shutil,hashlib
p=Path('outputs/PlateVision1');out=p/'models/paddle_plate_adapted_v1';out.mkdir(exist_ok=True)
shutil.copy2(p/'models/paddle_head_half_trial_v1/english.onnx',out/'english.onnx')
for n in ['text_detector.onnx','RAPIDOCR_LICENSE.txt','PADDLEOCR_LICENSE.txt']:
 shutil.copy2(p/'models/paddle_candidate'/n,out/n)
report={'method':'Pretrained PP-OCRv5 English recognizer with locally adapted final CTC linear layer only; 50% original / 50% epoch-1 adapted projection. All feature-extractor weights unchanged.','training_images':76,'development_line_images':26,'full_pipeline_images':45,'full_plate_exact':31,'accepted_correct':27,'accepted_wrong':2,'threshold':.95,'baseline':{'exact':31,'accepted_correct':25,'accepted_wrong':2},'source_manifest':json.loads((p/'models/paddle_candidate/manifest.json').read_text()),'english_sha256':hashlib.sha256((out/'english.onnx').read_bytes()).hexdigest(),'note':'Development-selected, no independent accuracy claim. Original pretrained model retained for rollback.'}
(out/'adaptation.json').write_text(json.dumps(report,indent=2));shutil.copy2(p/'models/paddle_head_trial_v1/report.json',out/'training_report.json');shutil.copy2(p/'models/paddle_head_half_trial_v1/full_pipeline_evaluation.json',out/'development_evaluation.json')
