from pathlib import Path
import json,html
p=Path('outputs/PlateVision1/data/ocr_finetune_audit_v1');rows=json.loads((p/'train.json').read_text());review=json.loads((p/'layout_review.json').read_text())
bad={14,22,47,52,77}
kept=[]
for i,r in enumerate(rows):
 if i in bad:review.append({'id':r['id'],'image':r['original_image'],'crop':r['image'],'text':r['text'],'reason':'Visual audit: detected line crop omits characters from the full label'})
 else:kept.append(r)
(p/'train.json').write_text(json.dumps(kept,indent=2));(p/'train.txt').write_text(''.join(f"images/{r['id']}.png\t{r['text']}\n" for r in kept),encoding='utf8');(p/'layout_review.json').write_text(json.dumps(review,indent=2))
card=json.loads((p/'audit.json').read_text());card['counts']['single_line_after_visual_screen']=len(kept);card['counts']['incomplete_line_crops_quarantined']=len(bad);card['counts']['total_layout_review']=len(review);card['training_started']=False
(p/'audit.json').write_text(json.dumps(card,indent=2))
page=['<!doctype html><meta charset="utf-8"><title>OCR training crop audit</title><style>body{background:#111;color:#eee;font:18px sans-serif;max-width:950px;margin:30px auto}article{border:1px solid #555;padding:15px;margin:20px 0}img{max-width:800px;min-height:65px;image-rendering:auto}code{color:#59fbd1}</style><h1>OCR training crop audit</h1><p>Read-only audit. These records are excluded from the initial training candidate. Original human labels are preserved. No training starts on this page.</p>']
for r in review:
 page.append('<article><code>'+html.escape(r['text'])+'</code><p>'+html.escape(r['reason'])+'</p><img src="'+Path(r['image']).as_uri()+'"><p>'+html.escape(r['id'])+'</p></article>')
(p/'REVIEW_LAYOUT.html').write_text('\n'.join(page),encoding='utf8')
print(json.dumps(card,indent=2))
