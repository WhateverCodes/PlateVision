from pathlib import Path
import json,shutil,zipfile
root=Path.cwd();p=root/'outputs/PlateVision1'
s=(root/'work/train_paddle_head.py').read_text(encoding='utf-8-sig')
s=s.replace("root=Path.cwd();sys.path.insert(0,str(root/'work/onnx_tools'));sys.path.insert(0,str(root/'outputs/PlateVision1'))", "root=Path(__file__).resolve().parents[1]")
s=s.replace("p=root/'outputs/PlateVision1';data=p/'data/ocr_finetune_audit_v1';out=p/'models/paddle_head_trial_v1'", "import argparse\nparser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()\np=root;data=p/'data/ocr_finetune_audit_v1';out=args.out")
s=s.replace("best_state[0] if tensor.name==weight_name else best_state[1]", "((best_state[0]+w0)*.5) if tensor.name==weight_name else ((best_state[1]+b0)*.5)")
s=s.replace("'selected_epoch':best_state[2] if best_state else None", "'selected_epoch':best_state[2] if best_state else None,'export_blend':0.5")
(p/'training/adapt_paddle_projection.py').write_text(s,encoding='utf8')
doc='''# PaddleOCR final-layer adaptation — 3 October 2026

The app now prefers models/paddle_plate_adapted_v1 when installed. Original models/paddle_candidate weights remain available for rollback. Our own CNN was not changed.

This bypasses the unavailable training-checkpoint download by extracting the final CTC projection from the existing pretrained ONNX graph. Its input features stay frozen. We verified that the extracted linear layer reproduces the original model probabilities, then trained that layer with PyTorch CTC loss on 76 screened human-labelled training lines. This is final-layer adaptation of pretrained PaddleOCR, not full-network fine-tuning and not training our own CNN.

The separate development line subset contains 26 crops. The baseline read 15 exactly; the adapted candidate read 16. Eight short epochs were evaluated; epoch 1 was the first best candidate. The unblended model increased wrong accepted readings in the complete pipeline, so a conservative 50% original / 50% adapted parameter update was evaluated and selected.

| Same 45 development plate crops | Before | Selected update |
|---|---:|---:|
| Complete plate exact | 31 (68.9%) | 31 (68.9%) |
| Correct accepted | 25 | 27 |
| Incorrect accepted | 2 | 2 |
| Rejected | 18 | 16 |

The confidence threshold remains 0.95; it was not lowered. This is improved acceptance on development data, not increased exact recognition accuracy or an independent 95% claim. The 26-line subset is part of these development images. Reserved final photos were not used. Pretrained overlap and capture independence remain uncertain.

16 targeted app tests passed, including real-model inference and comparison. The user's photos 6 and 8 remain accepted correctly; photos 3 and 10 still read correctly but remain uncertain. Image 4 still disagrees with the user's confirmed label. The vehicle classifier is unchanged.

Restart OPEN_PLATEVISION.cmd in your existing project folder. No download, training clicks or new character reviews are needed on this laptop.

Developer reproduction: install ONNX tooling in a separate training environment alongside the project's PyTorch, NumPy and OpenCV dependencies, then run python -m training.adapt_paddle_projection --out models/new_projection_trial. Requires the prepared local data/ocr_finetune_audit_v1 training and validation files, which are not included in the project ZIP. The script refuses to overwrite a trial directory, uses two CPU threads and a 120-second gradient-update budget, and exports a conservative blended recognizer only if line-level validation improves. Full pipeline evaluation is still required before deployment. Source hashes, model provenance and detailed results accompany the adapted model.
'''
(p/'docs/PADDLE_PROJECTION_UPDATE.md').write_text(doc,encoding='utf8')
card_path=p/'data/ocr_finetune_audit_v1/audit.json';card=json.loads(card_path.read_text());card.update(training_started=True,training_method='Frozen ONNX features, final CTC projection adaptation using PyTorch; official full-network checkpoint still unavailable');card_path.write_text(json.dumps(card,indent=2))
start=p/'START_HERE.md';start.write_text('# Latest: PaddleOCR final-layer update\n\nRestart OPEN_PLATEVISION.cmd. Correct accepted development readings increased from 25 to 27 out of 45; exact accuracy remains 68.9%. See docs/PADDLE_PROJECTION_UPDATE.md.\n\n'+start.read_text(encoding='utf8'),encoding='utf8')
files=['config.py','src/whole_line_reader.py','training/adapt_paddle_projection.py','docs/PADDLE_PROJECTION_UPDATE.md','START_HERE.md','review_glyphs.py','START_200_CHARACTER_REVIEW.cmd']+[str(f.relative_to(p)).replace('\\','/') for f in (p/'models/paddle_plate_adapted_v1').iterdir() if f.is_file()]
for t in (root/'outputs').glob('PlateVision*'):
 if t==p or not (t/'app.py').exists():continue
 for f in files:
  (t/f).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p/f,t/f)
 print('Updated',t.name)
base=root/'outputs/PLATEVISION_CNN_Update.zip';out=root/'outputs/PLATEVISION_Paddle_Update_2026-10-03.zip'
with zipfile.ZipFile(base) as old:
 prefix=next(n[:-6] for n in old.namelist() if n.endswith('/app.py') and n.count('/')==1)
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as new:
  for item in old.infolist():
   if item.filename not in {prefix+f for f in files}:new.writestr(item,old.read(item.filename))
  for f in files:new.write(p/f,prefix+f)
with zipfile.ZipFile(out) as z:assert z.testzip() is None
print('ZIP verified:',out)
