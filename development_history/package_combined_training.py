import hashlib,json,shutil,zipfile
from pathlib import Path
p=Path('outputs/PlateVision1')
files=[Path(x) for x in ['config.py','src/whole_line_reader.py','README.md','START_HERE.md','docs/COMBINED_DATASET_TRAINING.md','outputs/evaluation/character_metrics.json','outputs/evaluation/combined_training_summary.json','outputs/evaluation/combined_new_validation.json','outputs/evaluation/combined_detector_before.json','outputs/evaluation/combined_detector_after.json','outputs/evaluation/combined_detector_full_epoch.json']]
files+=list(f.relative_to(p) for f in (p/'training').glob('*combined*.py'))
files.append(Path('tests/test_whole_line.py'))
for directory in ['models/paddle_combined_v2','models/plate_detector_real_v3']:
    files+=list(f.relative_to(p) for f in (p/directory).iterdir() if f.is_file())
files += [Path('models/combined_cnn_2026_10_03/epoch2.pt'),Path('models/combined_cnn_2026_10_03/report.json')]
assert all((p/f).is_file() for f in files)
copies=[d for d in Path('outputs').glob('PlateVision*') if d.is_dir() and (d/'app.py').exists() and d!=p]
for dest in copies:
    for relative in files:
        target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p/relative,target)
base=Path('outputs/PLATEVISION_Simplified_2026-10-03.zip');out=Path('outputs/PLATEVISION_Trained_Datasets_2026-10-04.zip')
with zipfile.ZipFile(base) as z:
    prefix=next(n[:-len('app.py')] for n in z.namelist() if n.endswith('/app.py') and n.count('/')==1)
    replacements={prefix+f.as_posix():p/f for f in files}
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as target:
        for info in z.infolist():
            if info.filename not in replacements:target.writestr(info,z.read(info.filename))
        for name,source in replacements.items():target.write(source,name)
with zipfile.ZipFile(out) as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(set(z.namelist()))
    for name,source in replacements.items():assert hashlib.sha256(z.read(name)).digest()==hashlib.sha256(source.read_bytes()).digest()
for dest in copies:
    for relative in files:assert hashlib.sha256((dest/relative).read_bytes()).digest()==hashlib.sha256((p/relative).read_bytes()).digest()
print(json.dumps(dict(zip=str(out.resolve()),bytes=out.stat().st_size,synchronized=[d.name for d in copies],verified_files=len(files)),indent=2))
