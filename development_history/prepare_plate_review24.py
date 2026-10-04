from pathlib import Path
import json,html,hashlib
p=Path('outputs/PlateVision1/data/dataset_trial_2026_10_03');rows=json.loads((p/'plate_candidates.json').read_text());rank=sorted(rows,key=lambda r:(r['height']>=24,r['width']>=80,r['height']),reverse=True)[:24]
parts=['<!doctype html><meta charset="utf-8"><title>PLATEVISION — new plate label review</title><style>body{background:#101419;color:#eee;font:18px sans-serif;padding:24px}section{display:inline-block;vertical-align:top;width:30%;margin:1%;border:1px solid #567;padding:10px;box-sizing:border-box}img{max-width:100%;height:85px;object-fit:contain}b{color:#62ffe0}</style><h1>24 new plate crops to check</h1><p>These are dataset-supplied labels, not verified answers. Check each plate against its text. This page does not save or train anything. Send corrections as “number → correct text”, or say which numbers are unreadable.</p>']
for i,r in enumerate(rank,1):
 r['review_number']=i;parts.append(f'<section><h3>{i}</h3><img src="plates/{Path(r["crop"]).name}"><p>Supplied: <b>{html.escape(r["text"])}</b></p></section>')
(p/'REVIEW_24_PLATES.html').write_text(''.join(parts),encoding='utf8');(p/'review_24.json').write_text(json.dumps(rank,indent=2))
a=Path('C:/Users/Grace/Downloads/archive (5).zip');b=Path('C:/Users/Grace/Downloads/archive (2).zip')
print('Archive 5 identical ZIP SHA256 to archive 2:',hashlib.file_digest(a.open('rb'),'sha256').hexdigest()==hashlib.file_digest(b.open('rb'),'sha256').hexdigest())
