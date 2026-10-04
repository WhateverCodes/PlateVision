from pathlib import Path
import json,hashlib,collections,cv2,sys
root=Path(__file__).resolve().parents[1];p=root/'outputs/PlateVision1';sys.path.insert(0,str(p))
from src.whole_line_reader import WholeLineReader
out=p/'data/ocr_finetune_audit_v1';out.mkdir(exist_ok=True)
train=json.loads((p/'data/reader_reviewed_v2/train.json').read_text())
val=json.loads((p/'data/reader_reviewed_v2/val.json').read_text())+json.loads((p/'data/accuracy_check_2026_09_21/val.json').read_text())
test=json.loads((p/'data/reader_reviewed_v2/test.json').read_text())
reserved=json.loads((p/'outputs/external_test/reserved_test_manifest.json').read_text())['records']
held=val+test+reserved
held_text={t for r in held for t in r.get('texts',[r.get('text','')]) if t}
held_hash={r.get('sha256') for r in held};held_group={r.get('group') for r in held if r.get('group')}
reader=WholeLineReader(p/'models/paddle_candidate');counts=collections.Counter();accepted=[];review=[];seen=set()
for r in train:
 if r.get('label_origin')!='human_reviewed':counts['unverified_excluded']+=1;continue
 f=Path(r['image']);data=f.read_bytes();digest=hashlib.sha256(data).hexdigest()
 if r['text'] in held_text or digest in held_hash or r.get('group') in held_group:
  counts['holdout_conflict_excluded']+=1;review.append({'id':r['id'],'reason':'held-out identity overlap'});continue
 if digest in seen:counts['exact_duplicate_excluded']+=1;continue
 seen.add(digest);im=cv2.imread(str(f));lines=reader.line_crops(im)
 if len(lines)!=1:
  counts['line_layout_review']+=1;review.append({'id':r['id'],'image':str(f),'text':r['text'],'reason':f'{len(lines)} detected text regions; full-plate label cannot safely label each line'});continue
 crop=lines[0]['crop'];dest=out/'images'/f"{r['id']}.png";dest.parent.mkdir(exist_ok=True);cv2.imwrite(str(dest),crop)
 accepted.append({**r,'image':str(dest),'original_image':str(f),'original_sha256':digest,'line_crop_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
 counts['verified_single_line_training']+=1
 if len(accepted)%20==0:print('Prepared',len(accepted),'verified single-line examples',flush=True)
(out/'train.json').write_text(json.dumps(accepted,indent=2))
(out/'train.txt').write_text(''.join(f"images/{r['id']}.png\t{r['text']}\n" for r in accepted),encoding='utf8')
(out/'layout_review.json').write_text(json.dumps(review,indent=2))
card={'counts':dict(counts),'existing_development_images':len(val),'existing_test_manifest_records':len(test),'reserved_final_manifest_records':len(reserved),'notes':['Only existing human-reviewed training labels used. No held-out images used for training.','Historical split membership preserved; no new independent test split claimed.','Exact hashes, registration labels and existing group identifiers screened. Near-duplicate capture independence still requires audit.','Single detected text region is not proof of complete text: exported crops need visual completeness audit before training.','App weights unchanged. This is data preparation, not fine-tuning.']}
(out/'audit.json').write_text(json.dumps(card,indent=2));print(json.dumps(card,indent=2))
