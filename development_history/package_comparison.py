from pathlib import Path
import json
import shutil
import zipfile

base=Path(__file__).resolve().parents[1]/'outputs'
root=base/'PlateVision1'
changed=['app.py','config.py','requirements.txt','SETUP_PRETRAINED.cmd','START_HERE.md',
 'src/pretrained.py','src/comparison.py','src/comparison_ui.py','src/pipeline.py',
 'src/character_segmenter.py','training/download_pretrained.py','training/evaluate_comparison.py',
 'training/improve_own_characters.py','tests/test_comparison.py','docs/MODEL_COMPARISON.md',
 'models/character_reviewed_augmented/report.json']
changed += ['review_glyphs.py','open_glyph_review.py','REVIEW_TRAINING_CHARACTERS.cmd','training/prepare_glyph_review.py','training/prepare_verified_characters.py','tests/test_glyph_review.py','docs/NEXT_CHARACTER_CHECK.md','src/character_classifier.py']
manifest=json.loads((root/'models/pretrained/manifest.json').read_text())
changed += ['models/pretrained/'+r['file'] for r in manifest]+['models/pretrained/manifest.json']
for name in ['PlateVision2','PlateVision3']:
    destination=(base/name).resolve()
    assert destination.parent==base.resolve()
    for rel in changed:
        target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/rel,target)

output=base/'PLATEVISION_Model_Comparison.zip'
previous=base/'PLATEVISION_Updated_2026-09-21.zip'
with zipfile.ZipFile(previous) as source:
    prefix=next(n for n in source.namelist() if n.endswith('/app.py'))[:-len('app.py')]
    replacements={prefix+rel:(root/rel).read_bytes() for rel in changed}
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as dest:
        for info in source.infolist():
            if info.filename not in replacements:dest.writestr(info,source.read(info.filename))
        for name,content in replacements.items():dest.writestr(name,content)
with zipfile.ZipFile(output) as check:
    assert check.testzip() is None
    assert check.read(prefix+'src/comparison.py')==(root/'src/comparison.py').read_bytes()
print('Updated all three local project copies.')
print(str(output),round(output.stat().st_size/1024/1024,1),'MB; ZIP integrity passed.')
