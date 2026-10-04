from pathlib import Path
import zipfile, shutil, json
root=Path.cwd(); project=root/'outputs/PlateVision1'
doc='''# OCR refinement - 2 October 2026

Restart OPEN_PLATEVISION.cmd. In Recognition choose Whole-line OCR (pretrained) and Pretrained plate detector. Keep reading confidence at 0.95. No training or further character reviews are needed for this update.

The reader now compensates for slanted text ordering and considers alternative letters/digits actually suggested by the OCR network when its first reading fails the existing registration format. Valid first readings are preserved. Candidate ranking includes spaces even when removed from the displayed registration. Selected token scores are not increased. This is bounded token-alternative decoding, not full CTC beam search or newly trained weights.

| Reviewed development crops | Previous exact | Updated exact | Updated accepted correct | Updated accepted wrong |
|---|---:|---:|---:|---:|
| Newer set, 30 | 14 | 18 | 13 | 2 |
| Difficult set, 15 | 10 | 13 | 8 | 0 |
| Total, 45 | 24 (53.3%) | 31 (68.9%) | 21 | 2 |

Exact counts include uncertain readings shown in inspection results. At the default confidence threshold, only 21/45 are both correct and accepted; 22/45 are rejected and 2/45 incorrectly accepted. A 0.95 confidence setting does not mean 95% accuracy. The previous version accepted 19 correct and 2 wrong. Two remaining confidently wrong readings are MH01AV7467 instead of MH01AV7461, and WH02BJ2456 instead of MH02BJ2456.

These small crop sets have been used for development; this is not independent final-test or full-photo accuracy. External pretrained data overlap is unknown. Reserved final photographs remain unused. No own CNN weights or training data were changed. All inference stays local with two CPU threads.

Validation: 14 targeted tests passed, including the actual app demonstration, comparison, pipeline, slanted/two-row ordering, repeated CTC characters, separator ranking, and preservation of alternative token scores.

Your next step: restart the app and try 10 new, clear vehicle photographs in Recognition. Keep the full original photo and record its correct plate text and the app result, including failures. Send these together in a ZIP for diagnosis. Use photos separate from the reserved final test set. Do not crop away vehicles for this check; it needs to expose plate detection as well as reading errors. More individual-character review is not needed for the pretrained OCR route.
'''
(project/'docs/OCR_REFINEMENT.md').write_text(doc,encoding='utf-8')
p=project/'START_HERE.md'; s=p.read_text(encoding='utf-8'); p.write_text('# Latest OCR refinement\n\nRestart OPEN_PLATEVISION.cmd. OCR now reads 31/45 development crops exactly, up from 24/45. See docs/OCR_REFINEMENT.md for acceptance rates and your next step. No retraining needed.\n\n'+s,encoding='utf-8')
p=project/'docs/WHOLE_LINE_UPDATE.md'; s=p.read_text(encoding='utf-8'); p.write_text('> Historical first-upgrade results below. For current OCR results and instructions, see OCR_REFINEMENT.md.\n\n'+s,encoding='utf-8')
files=['src/whole_line_reader.py','src/paddle_candidate.py','tests/test_whole_line.py','docs/OCR_REFINEMENT.md','docs/WHOLE_LINE_UPDATE.md','START_HERE.md','outputs/evaluation/ocr_refined_30.json','outputs/evaluation/ocr_refined_15.json']
for target in (root/'outputs').glob('PlateVision*'):
    if target==project or not (target/'app.py').exists():continue
    for name in files:
        dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(project/name,dest)
    print('Updated',target.name)
base=root/'outputs/PLATEVISION_Updated_Models_2026-10-02.zip'
out=root/'outputs/PLATEVISION_OCR_Refined_2026-10-02.zip'
with zipfile.ZipFile(base) as old:
    prefix=next(n[:-len('app.py')] for n in old.namelist() if n.endswith('/app.py') and n.count('/')==1)
    replacements={prefix+n for n in files}
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as new:
        for item in old.infolist():
            if item.filename not in replacements:new.writestr(item,old.read(item.filename))
        for name in files:new.write(project/name,prefix+name)
with zipfile.ZipFile(out) as z:assert z.testzip() is None
print('Verified ZIP',out,round(out.stat().st_size/1024**2,1),'MiB')
