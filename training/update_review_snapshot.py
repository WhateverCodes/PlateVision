"""Update labels while preserving previous held-out membership and training exposure."""
import json
from pathlib import Path
from collections import Counter
from src.dataset_review import load_reviews
from training.data_tools import save_json


def prepare(root):
    root = Path(root).resolve()
    folder = root / 'data/ocr_text_review'
    previous = root / 'data/reader_v1'
    out = root / 'data/reader_reviewed_v2'
    if out.exists():
        raise ValueError('Snapshot already exists; use a new version.')
    catalog = {r['id']: r for r in json.loads((folder/'catalog.json').read_text())}
    reviews = load_reviews(folder)
    old = {name: json.loads((previous/(name+'.json')).read_text())
           for name in ['train', 'val', 'test', 'withheld_unverified']}
    historical = {r['id']: r for rows in old.values() for r in rows}
    records = {i: {**r, **reviews.get(i, {})} for i, r in catalog.items()}
    # Connect historical groups, original labels, corrections and exact duplicate pixels.
    parent = {i: i for i in records}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, r in records.items():
        keys = [('source', catalog[i]['group']), ('pixels', r['sha256'])]
        keys += [('text', t) for t in [r['text'], catalog[i]['text']] if t]
        if i in historical:
            keys += [('historical', historical[i]['group'])]
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    train_groups = {find(r['id']) for r in old['train']}
    held_groups = {find(r['id']) for name in ['val','test','withheld_unverified'] for r in old[name]}
    conflict = train_groups & held_groups
    if conflict:
        raise ValueError('Corrected labels connect previously separate train/holdout groups; audit required.')
    reserved = json.loads((root/'outputs/external_test/reserved_test_manifest.json').read_text())['records']
    reserved_text = {t for r in reserved for t in r['texts']}
    reserved_hash = {r['sha256'] for r in reserved}
    reserved_groups = {find(i) for i,r in records.items() if r['text'] in reserved_text or r['sha256'] in reserved_hash}
    def row(i):
        r = records[i]
        path = (folder/r['image']).resolve()
        assert path.exists(), path
        return {**r, 'image':str(path), 'texts':[r['text']], 'group':find(i),
                'label_origin':'human_reviewed' if r['status']=='reviewed' else 'source_unverified'}
    train_ids = {r['id'] for r in old['train']}
    old_ids = set(historical)
    new_train = {i for i,r in reviews.items() if i not in old_ids and r['status']=='reviewed' and find(i) not in held_groups}
    train = [row(i) for i in sorted(train_ids | new_train)
             if not records[i]['status'].startswith('excluded') and find(i) not in reserved_groups]
    # Preserve the original evaluation records and labels, repairing only renamed paths.
    for name in ['val', 'test']:
        rows = [{**r, 'image':str((folder/catalog[r['id']]['image']).resolve())} for r in old[name]]
        save_json(out/(name+'.json'), rows)
    supplemental = [row(r['id']) for r in old['withheld_unverified']
                    if r['id'] in reviews and records[r['id']]['status']=='reviewed'
                    and find(r['id']) not in train_groups | reserved_groups]
    save_json(out/'train.json', train)
    save_json(out/'supplemental_val.json', supplemental)
    card = {'review_count':len(reviews), 'training_images':len(train),
            'training_origins':dict(Counter(r['label_origin'] for r in train)),
            'new_training_images':len(new_train), 'original_val':len(old['val']),
            'original_test':len(old['test']), 'supplemental_validation':len(supplemental),
            'note':'Original validation and test membership/labels preserved. Supplemental development validation uses newly reviewed, previously withheld images. Historical training exposure remains excluded even if labels change. Capture independence is not certified; previous near-duplicate groups preserved. External final photos untouched.'}
    save_json(out/'dataset_card.json', card)
    print(json.dumps(card, indent=2))


if __name__ == '__main__':
    prepare(Path(__file__).resolve().parents[1])
