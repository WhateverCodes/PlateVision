"""Export selected research detector and reader evidence, then synchronize app copies."""
import json,shutil,hashlib
from pathlib import Path
import torch
p=Path('outputs/PlateVision1')
read=lambda f:json.loads((p/f).read_text())
before=read('outputs/evaluation/combined_detector_before.json')
focused=read('outputs/evaluation/combined_detector_after.json')
full=read('outputs/evaluation/combined_detector_full_epoch.json')
options=[('models/combined_detector_indian_2026_10_03/best.pt',focused['models']['own']),('models/combined_detector_2026_10_03/best.pt',full['models']['own'])]
selected,metric=max(options,key=lambda pair:pair[1]['ap'])
checkpoint=torch.load(p/selected,map_location='cpu',weights_only=True)
dest=p/'models/plate_detector_real_v3';dest.mkdir(exist_ok=True)
slim={k:checkpoint[k] for k in ['format_version','kind','model','image_size','alphabet','trained_steps','provenance']}
slim['provenance']+=' | Selected on 61 Indian development images by AP@0.5. Not independent-test validated. Main app retains external YOLO detector.'
torch.save(slim,dest/'best.pt')
(dest/'evaluation.json').write_text(json.dumps(dict(selected_source=selected,metrics=metric,baseline=before['models']['own'],deployed_external=before['models']['deployed_yolo']),indent=2))
fullck=torch.load(p/'models/combined_detector_2026_10_03/last.pt',map_location='cpu',weights_only=True)
assert fullck['epoch']==1 and fullck['cursor']==0,'Mixed detector pass must complete before finalizing'
summary=dict(date='2026-10-04',scope='Development evaluations only. See COMBINED_DATASET_TRAINING.md for limitations.',dataset_ledger=read('data/combined_training_2026_10_03/dataset_ledger.json'),data_counts=read('data/combined_training_2026_10_03/audit.json'),reader_preparation=read('data/combined_training_2026_10_03/reader/audit.json'),detector=dict(baseline=before,indian_focused=focused,full_epoch=full,selected_custom_checkpoint=selected,selected_metrics=metric,mixed_epoch_complete=True,mixed_training_images=4238,mixed_steps=fullck['trained_steps'],main_app_detector='Retained external YOLO'),cnn=read('models/combined_cnn_2026_10_03/report.json'),ocr=read('models/paddle_combined_v2/development_evaluation.json'),additional_validation=read('outputs/evaluation/combined_new_validation.json'))
(p/'outputs/evaluation/combined_training_summary.json').write_text(json.dumps(summary,indent=2))
metrics=read('outputs/evaluation/character_metrics.json')
metrics['metrics']['paddleocr'].update(edit_errors=18,character_error_rate=18/440,character_score_1_minus_CER=1-18/440)
metrics['decision']='Retain previous CNN after mixed combined-data results; promote combined-data PaddleOCR at .91. See combined_training_summary.json.'
metrics['decoding_update']='Historical format-aware update is followed by the 2026-10-04 CTC projection training update.'
(p/'outputs/evaluation/character_metrics.json').write_text(json.dumps(metrics,indent=2))
doc=p/'docs/COMBINED_DATASET_TRAINING.md'
text=doc.read_text();text+='\n## Completed detector run\n\n'
text+='The mixed-data detector completed one full epoch over all 4,238 training images. A separate Indian-focused candidate completed one epoch over all 325 Indian training images after initialization from the earlier partial mixed pass. Both candidates were evaluated; the custom checkpoint with higher development AP@0.5 was retained for research.\n\n'
text+='| Detector on 61 development images / 66 labelled plates | Correct detections | False detections | Recall | AP@0.5 |\n|---|---:|---:|---:|---:|\n'
for name,m in [('Original own detector',before['models']['own']),('Indian-focused candidate',focused['models']['own']),('Full mixed epoch candidate',full['models']['own']),('Retained external YOLO',before['models']['deployed_yolo'])]:
    text+=f"| {name} | {m['true_positives']} | {m['false_positives']} | {m['recall']:.1%} | {m['ap']:.1%} |\n"
text+='\nDetection uses score .5 and IoU .5. Source boxes are not fully reviewed; these numbers are not a fully annotated independent test. The strongest custom detector is saved at `models/plate_detector_real_v3/best.pt`; main Recognition and Compare continue using YOLO.\n'
doc.write_text(text)
config=p/'config.py';text=config.read_text();text=text.replace('detector_path: Path = ROOT / "models/plate_detector/best.pt"','detector_path: Path = ROOT / ("models/plate_detector_real_v3/best.pt" if (ROOT / "models/plate_detector_real_v3/best.pt").exists() else "models/plate_detector/best.pt")');config.write_text(text)
print('Selected custom detector:',selected,metric)
