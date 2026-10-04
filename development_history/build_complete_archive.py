"""Create a full project archive without touching the frozen working project."""
from pathlib import Path
import os,json,shutil,hashlib,zipfile,csv,time

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/PlateVision1'
DEST=ROOT/'work/complete_archive_staging/Platevision'
ZIP=ROOT/'outputs/Platevision.zip'
assert not ZIP.exists(), 'Inspect existing output before replacing it'
DEST.mkdir(parents=True,exist_ok=True)
EXCLUDE={'.venv','venv','__pycache__','.pytest_cache','.git','.idea','node_modules'}
excluded=[]
def copy_tree(source,target):
    for directory,dirs,files in os.walk(source):
        for name in list(dirs):
            if name in EXCLUDE:
                excluded.append(str(Path(directory)/name));dirs.remove(name)
        relative=Path(directory).relative_to(source)
        for name in files:
            if name.endswith(('.pyc','.pyo')) or name in {'.env','secrets.toml'}:
                excluded.append(str(Path(directory)/name));continue
            out=target/relative/name;out.parent.mkdir(parents=True,exist_ok=True)
            original=Path(directory)/name
            if not out.exists() or original.stat().st_size!=out.stat().st_size or original.stat().st_mtime_ns!=out.stat().st_mtime_ns:
                shutil.copy2(original,out)

copy_tree(SOURCE,DEST)
print('Copied complete canonical project',flush=True)
# Include the exact latest final app rather than any superseded copies.
for name in ['app.py','config.py','src/whole_line_reader.py']:
    assert (SOURCE/name).read_bytes()==(ROOT/'outputs/PLATEVISION_FINAL'/name).read_bytes(),name

sources=[]
for name in ['3113449.zip']+[f'archive ({i}).zip' for i in [2,3,4,5,6,7,8,9,10]]:
    source=Path('C:/Users/Grace/Downloads')/name
    entry={'original_name':name,'available':source.exists()}
    if source.exists():
        target=DEST/'datasets/source_archives'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target);entry.update(archive_path=target.relative_to(DEST).as_posix(),bytes=target.stat().st_size)
        print('Included original dataset',name,flush=True)
    sources.append(entry)
copy_tree(ROOT/'outputs/Indian_Dataset_Starter',DEST/'datasets/Indian_Dataset_Starter')

# Development code and experiment artefacts, excluding disposable test output and environments.
history=DEST/'development_history';history.mkdir(exist_ok=True)
for f in (ROOT/'work').iterdir():
    if f.is_file() and f.suffix.lower() in {'.py','.json','.yaml','.yml','.jpg','.png','.txt','.log'}:
        shutil.copy2(f,history/f.name)
for name in ['budget_detector','evaluation_characters','evaluation_detector','evaluation_detector_positive','evaluation_pipeline','evaluation_pipeline_positive','format_baseline','new_dataset_audit','user_photo_batch','heldout_smoke','smoke_characters','smoke_detector','smoke_data','onnx_wheels','pretrained_wheels','paddle_downloads']:
    source=ROOT/'work'/name
    if source.exists():copy_tree(source,history/name)
print('Included development scripts and experiment artefacts',flush=True)

verification=DEST/'verification_confirmation';verification.mkdir(exist_ok=True)
for name in ['glyph_review','ocr_text_review','kaggle_indian_review','ocr_finetune_audit_v1','detection_archive10_review']:
    copy_tree(SOURCE/'data'/name,verification/name)
copy_tree(SOURCE/'outputs/dataset_review',verification/'dataset_review_reports')
for f in SOURCE.glob('*review*.py'):shutil.copy2(f,verification/f.name)
for f in SOURCE.glob('*REVIEW*.cmd'):shutil.copy2(f,verification/f.name)
for name in ['character_verified_100','character_verified_300']:
    copy_tree(SOURCE/'data'/name,verification/'cleaned_training_snapshots'/name)

def original_file(value):
    value=str(value).replace('\\','/')
    old=str(ROOT).replace('\\','/')+'/outputs/'
    if value.startswith(old):
        tail=value[len(old):].split('/',1)
        if len(tail)==2 and tail[0].lower().startswith('platevision'):
            return SOURCE/tail[1]
    return Path(value)

reviews=[]
for f in sorted((SOURCE/'data/glyph_review/reviews').glob('*.json')):
    r=json.loads(f.read_text());image=original_file(r['image']);target=verification/'character_crops_300'/f'{f.stem}{image.suffix}'
    assert image.is_file(),image
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(image,target)
    reviews.append(dict(id=f.stem,label=r.get('label',''),unusable=r.get('unusable',False),suggested=r.get('suggested',''),plate_text=r.get('plate_text',''),character_crop=target.relative_to(DEST).as_posix(),record='verification_confirmation/glyph_review/reviews/'+f.name))
with (verification/'CHARACTER_REVIEW_300.csv').open('w',newline='',encoding='utf-8-sig') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(reviews[0]));writer.writeheader();writer.writerows(reviews)
counts={'characters_checked':len(reviews),'usable_characters':sum(not r['unusable'] for r in reviews),'excluded_characters':sum(bool(r['unusable']) for r in reviews),'ocr_text_reviews':len(list((SOURCE/'data/ocr_text_review/reviews').glob('*.json'))),'dataset_reviews':len(list((SOURCE/'data/kaggle_indian_review/reviews').glob('*.json')))}
assert counts['characters_checked']==300 and counts['usable_characters']==299
(verification/'REVIEW_COUNTS.json').write_text(json.dumps(counts,indent=2))
(verification/'README.md').write_text('''# Verification and confirmation records

This folder collects the manual review evidence separately from the final application.

- `CHARACTER_REVIEW_300.csv`: an index of all 300 saved character decisions, including 299 usable characters and one excluded crop.
- `character_crops_300/`: the 300 corresponding character images.
- `glyph_review/`: original review records, queue and the 200-character follow-up batch.
- `ocr_text_review/`: 200 plate-text review records and their catalogue.
- `kaggle_indian_review/`: 11 dataset review records and the source catalogue.
- `cleaned_training_snapshots/`: the verified-100 and verified-300 training dataset versions.
- Other folders contain source-data audits, contact sheets, and cleaning reports.

These are preserved records, not a request to review the data again. Working copies remain under the main `data/` folder because existing training and review tools expect those locations. This deliberate duplication keeps the frozen application and scripts intact. Review-program copies here are for reference; use the launchers in the project root when running review tools.
''',encoding='utf-8')
(DEST/'datasets/SOURCE_ARCHIVES.json').write_text(json.dumps(sources,indent=2))

# A separate optional utility makes historical absolute manifest paths relocatable.
helper=r'''"""Relink archived JSON paths after extraction; never trains or alters model weights."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
OLD="C:/Users/Grace/Documents/Codex/2026-09-19/i-n"
state=ROOT/'.archive_path_state.json'
previous=json.loads(state.read_text()).get('root') if state.exists() else None
pairs=[]
for name in ['PlateVision']+[f'PlateVision{i}' for i in range(1,10)]+['PLATEVISION_FINAL']:
    pairs.append((OLD+'/outputs/'+name,str(ROOT)))
pairs += [(OLD+'/outputs/Indian_Dataset_Starter',str(ROOT/'datasets/Indian_Dataset_Starter')),(OLD+'/work',str(ROOT/'development_history')),('C:/Users/Grace/Downloads',str(ROOT/'datasets/source_archives'))]
if previous and previous!=str(ROOT):pairs.insert(0,(previous,str(ROOT)))
pairs.sort(key=lambda pair:len(pair[0]),reverse=True)
def transform(value):
    if isinstance(value,dict):return {k:transform(v) for k,v in value.items()}
    if isinstance(value,list):return [transform(v) for v in value]
    if isinstance(value,str):
        normalized=value.replace('\\','/')
        here=str(ROOT).replace('\\','/')
        if normalized.lower()==here.lower() or normalized.lower().startswith(here.lower()+'/'):
            return value
        for old,new in pairs:
            old=old.replace('\\','/')
            if normalized.lower()==old.lower() or normalized.lower().startswith(old.lower()+'/'):
                return str(Path(new)/normalized[len(old):].lstrip('/'))
    return value
changed=0
for directory in ['data','outputs','verification_confirmation']:
    for path in (ROOT/directory).rglob('*.json'):
        if 'original_path_manifests' in path.parts:continue
        try:
            original=json.loads(path.read_text(encoding='utf-8-sig'));updated=transform(original)
        except (ValueError,UnicodeError):continue
        if original!=updated:
            backup=ROOT/'verification_confirmation/original_path_manifests'/path.relative_to(ROOT)
            if not backup.exists():
                backup.parent.mkdir(parents=True,exist_ok=True);backup.write_bytes(path.read_bytes())
            path.write_text(json.dumps(updated,indent=2),encoding='utf-8');changed+=1
for filename,folder in [('glyph_review_location.txt','glyph_review'),('ocr_review_location.txt','ocr_text_review')]:
    (ROOT/filename).write_text(str(ROOT/'data'/folder),encoding='utf-8')
state.write_text(json.dumps({'root':str(ROOT)},indent=2))
print(f'Relinked {changed} JSON files to {ROOT}. No training started. Original manifests are preserved.')
'''
(DEST/'PREPARE_DATA_PATHS.py').write_text(helper,encoding='utf-8')
(DEST/'PREPARE_DATA_PATHS.cmd').write_text(r'''@echo off
setlocal
cd /d "%~dp0"
if exist "..\..\work\venv\Scripts\python.exe" (
    "..\..\work\venv\Scripts\python.exe" PREPARE_DATA_PATHS.py
) else if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" PREPARE_DATA_PATHS.py
) else (
    py -3 PREPARE_DATA_PATHS.py
)
pause
endlocal
''')

notes='''# Platevision — complete final project archive

## Start here

Extract the ZIP completely before running it. Close any older running PLATEVISION command window, then open `OPEN_PLATEVISION.cmd` in this folder. Keep its command window open. The model selection and weights are the finalized version; this archive does not retrain or change the models.

On the original laptop, extracting `Platevision/` directly into the project's `outputs/` folder lets the launcher use the existing Python environment. On another computer, follow the Python installation steps in `README.md`.

Before using the archived datasets for review or training after extraction, run `PREPARE_DATA_PATHS.cmd` once. Historical manifests contain absolute paths from the original laptop; this utility updates those paths and preserves copies of changed originals in `verification_confirmation/original_path_manifests/`. Run it again if you move the folder. It does not download anything or start training. Historical resumable checkpoints can remain tied to the original manifest fingerprint; do not bypass their mismatch checks. Reinitialize a new training run from saved weights when needed.

## Contents

- Final app, complete project source, training/evaluation scripts, tests, notebooks, README, documentation and licences.
- `models/`: selected weights, earlier models, experimental checkpoints, optimizer checkpoints where originally saved, and training reports.
- `data/`: all prepared project data, train/validation/test manifests, character crops, synthetic data, plate images and source reviews.
- `datasets/source_archives/`: all nine available original supplied dataset ZIPs. Duplicate source archives are retained for completeness, not counted as additional training data.
- `datasets/Indian_Dataset_Starter/`: the original starter images, ZIP and notes.
- `verification_confirmation/`: the separate manual-verification and cleaning record, including all 300 character decisions and matching crops.
- `outputs/`: evaluation results, training logs, audits and reserved-test manifests.
- `development_history/`: development scripts, earlier experiment outputs, and downloaded dependency wheels. Historical scripts may contain original workspace paths and are preserved as records; do not run old deployment scripts against the final version. Wheel downloads are historical, not a replacement for the pinned runtime requirements.
- `ARCHIVE_MANIFEST.json`: file-by-file sizes and SHA-256 checksums (excluding the manifest itself).

## Availability and exclusions

The previously supplied `3113449.zip` is no longer present at its original location and was not found in Downloads. It cannot be included. `datasets/SOURCE_ARCHIVES.json` records this explicitly.

Python virtual environments, installed package caches, temporary pytest output, editor metadata, bytecode and Git internals are omitted. Dependencies are documented in `requirements.txt`; these machine-specific generated folders are not project datasets or source code. Repeated copies of the whole project and earlier delivery ZIPs are not nested into this archive.

This is a complete local project backup, including research and evaluation data. The metrics and model limitations are documented in `docs/COMBINED_DATASET_TRAINING.md`.
'''
(DEST/'ARCHIVE_CONTENTS.md').write_text(notes,encoding='utf-8')
readme=DEST/'README.md';readme.write_text('# Complete archive\n\nSee [ARCHIVE_CONTENTS.md](ARCHIVE_CONTENTS.md) for the archive layout, dataset availability and extraction instructions. Manual checks are collected in [verification_confirmation](verification_confirmation/README.md).\n\n'+readme.read_text(),encoding='utf-8')
(DEST/'ARCHIVE_SOURCE_REPORT.json').write_text(json.dumps({'source_project':str(SOURCE),'final_app_verified_against':str(ROOT/'outputs/PLATEVISION_FINAL'),'missing':['3113449.zip'],'review_counts':counts,'excluded_generated_directories':sorted(EXCLUDE),'source_archives':sources},indent=2))
print('Staging complete',counts,flush=True)

manifest=[];files=sorted(f for f in DEST.rglob('*') if f.is_file() and f.name!='ARCHIVE_MANIFEST.json')
for index,f in enumerate(files):
    digest=hashlib.sha256()
    with f.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    manifest.append({'path':f.relative_to(DEST).as_posix(),'bytes':f.stat().st_size,'sha256':digest.hexdigest()})
    if index%15000==0:print('Inventoried',index,'/',len(files),flush=True)
(DEST/'ARCHIVE_MANIFEST.json').write_text(json.dumps({'file_count':len(files),'total_bytes':sum(r['bytes'] for r in manifest),'files':manifest},indent=2))
files.append(DEST/'ARCHIVE_MANIFEST.json')
partial=ZIP.with_suffix('.zip.partial')
with zipfile.ZipFile(partial,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
    for index,f in enumerate(files):
        compression=zipfile.ZIP_STORED if f.suffix.lower() in {'.zip','.whl','.jpg','.jpeg','.png'} else zipfile.ZIP_DEFLATED
        archive.write(f,'Platevision/'+f.relative_to(DEST).as_posix(),compress_type=compression,compresslevel=1)
        if index%15000==0:print('Archived',index,'/',len(files),flush=True)
partial.replace(ZIP)
print('ZIP written',ZIP,ZIP.stat().st_size,flush=True)
with zipfile.ZipFile(ZIP) as archive:
    assert len(archive.namelist())==len(set(archive.namelist()))
    for index,r in enumerate(manifest):
        digest=hashlib.sha256()
        with archive.open('Platevision/'+r['path']) as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        assert digest.hexdigest()==r['sha256'],r['path']
        if index%15000==0:print('Verified',index,'/',len(manifest),flush=True)
digest=hashlib.sha256()
with ZIP.open('rb') as stream:
    for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
ZIP.with_suffix('.sha256').write_text(digest.hexdigest()+'  Platevision.zip\n')
report={'zip':str(ZIP),'bytes':ZIP.stat().st_size,'files':len(files),'review_counts':counts,'all_file_hashes_verified':True,'sha256':digest.hexdigest(),'missing':['3113449.zip']}
(ROOT/'outputs/Platevision_archive_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
