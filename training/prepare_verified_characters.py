"""Apply human glyph corrections and exclusions to a NEW training snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
from config import ROOT,ALPHABET
from training.data_tools import save_json


def prepare(root,out):
    root,out=Path(root),Path(out)
    if out.exists():raise ValueError('Use a new dataset directory; preserve prior experiments.')
    queue={r['id']:r for r in json.loads((root/'data/glyph_review/queue.json').read_text())}
    reviews={p.stem:json.loads(p.read_text()) for p in (root/'data/glyph_review/reviews').glob('*.json')}
    if len(reviews)<50:raise ValueError('Finish at least 50 character checks before preparing this experiment.')
    data=root/'data/character_adaptation_v2'
    rows=json.loads((data/'train.json').read_text());result=[];removed=0;verified=0
    allowed={r['id'] for r in json.loads((root/'data/reader_reviewed_v2/train.json').read_text())}
    held={r['id'] for name in ['val','test','withheld_unverified']
          for r in json.loads((root/'data/reader_v1'/f'{name}.json').read_text())}
    for row in rows:
        token=Path(row['image']).stem
        if row['origin']=='synthetic':result.append(row);continue
        source=token.rsplit('_',1)[0]
        if source not in allowed or source in held:raise ValueError('Training/evaluation overlap detected.')
        if token not in reviews:
            result.append(row);continue
        review=reviews[token];item=queue.get(token)
        if item is None or item['source_id']!=source or review['status']!='human_verified_glyph':
            raise ValueError('Invalid glyph review.')
        if hashlib.sha256(Path(row['image']).read_bytes()).hexdigest()!=item['image_sha256']:
            raise ValueError('Glyph image changed; check it again before training.')
        if review['unusable']:removed+=1;continue
        if review['label'] not in ALPHABET or len(review['label'])!=1:raise ValueError('Invalid character label.')
        verified+=1
        result.append({**row,'label':review['label'],'origin':'human_verified_glyph'})
    if not verified:raise ValueError('Need readable verified glyphs, not only excluded crops.')
    save_json(out/'train.json',result)
    card={'human_verified':verified,'excluded_bad_crops':removed,'characters':len(result),
          'note':'Human glyph corrections override weak labels; rejected crops excluded. Other automatic training alignments remain provisional. No evaluation images added.'}
    save_json(out/'dataset_card.json',card);print(json.dumps(card,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    prepare(ROOT,p.parse_args().out)
