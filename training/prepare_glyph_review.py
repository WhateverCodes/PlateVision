"""Prioritize uncertain/mismatched training glyphs; never review evaluation glyphs."""
import json
import hashlib
from pathlib import Path
import cv2
from config import ROOT
from src.runtime import configure
from src.character_classifier import Classifier
from src.character_segmenter import Character


def prepare():
    configure(2)
    folder=ROOT/'data/glyph_review'
    if (folder/'queue.json').exists():raise ValueError('Preserve existing review queue.')
    data=ROOT/'data/character_adaptation_v2'
    audit={r['id']:r for r in json.loads((data/'alignment_audit.json').read_text()) if r['label_origin']=='human_reviewed'}
    plates={r['id']:r for r in json.loads((ROOT/'data/reader_reviewed_v2/train.json').read_text())}
    held={r['id'] for name in ['val','test','withheld_unverified'] for r in json.loads((ROOT/'data/reader_v1'/f'{name}.json').read_text())}
    reader=Classifier(ROOT/'models/character_adaptation_v1/last.pt')
    queue=[]
    for row in json.loads((data/'train.json').read_text()):
        if row['origin']!='weak_alignment':continue
        token=Path(row['image']).stem;identity,index=token.rsplit('_',1)
        if identity not in audit:continue
        assert identity in plates and identity not in held
        glyph=cv2.imread(row['image'],0);pred,scores=reader([Character((0,0,32,32),glyph)])
        queue.append({'id':token,'source_id':identity,'index':int(index),'image':row['image'],
                      'plate_image':plates[identity]['image'],'plate_text':plates[identity]['text'],
                      'suggested':row['label'],'predicted':pred,'score':scores[0],
                      'group':plates[identity]['group'],'split':'train',
                      'image_sha256':hashlib.sha256(Path(row['image']).read_bytes()).hexdigest()})
    queue.sort(key=lambda r:(r['predicted']==r['suggested'],r['score'],r['id']))
    # Round-robin labels to avoid spending every review on common digits.
    buckets={}
    for r in queue:buckets.setdefault(r['suggested'],[]).append(r)
    ordered=[]
    while any(buckets.values()):
        for rows in buckets.values():
            if rows:ordered.append(rows.pop(0))
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'queue.json').write_text(json.dumps(ordered,indent=2))
    print(f'Prepared {len(ordered)} training-only glyphs; start with 50. Evaluation images excluded.')


if __name__=='__main__':prepare()
