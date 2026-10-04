from pathlib import Path
import shutil,zipfile

base=Path(__file__).resolve().parents[1]/'outputs';root=base/'PlateVision1'
changed=['app.py','config.py','START_HERE.md','SETUP_PRETRAINED.cmd',
 'src/pipeline.py','src/comparison.py','src/comparison_ui.py','src/paddle_candidate.py','src/whole_line_reader.py',
 'training/download_whole_line.py','training/evaluate_comparison.py','tests/test_whole_line.py','docs/WHOLE_LINE_UPDATE.md']
changed += [p.relative_to(root).as_posix() for p in (root/'models/paddle_candidate').iterdir() if p.is_file()]
for name in ['PlateVision2','PlateVision3','PlateVision4','PlateVision5']:
    target=base/name
    assert target.resolve().parent==base.resolve()
    for rel in changed:
        dest=target/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/rel,dest)
previous=base/'PLATEVISION_Model_Comparison.zip';output=base/'PLATEVISION_Whole_Line_OCR_2026-10-02.zip'
with zipfile.ZipFile(previous) as source:
    prefix=next(n for n in source.namelist() if n.endswith('/app.py'))[:-len('app.py')]
    updates={prefix+rel:(root/rel).read_bytes() for rel in changed}
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as dest:
        for info in source.infolist():
            if info.filename not in updates:dest.writestr(info,source.read(info.filename))
        for name,data in updates.items():dest.writestr(name,data)
with zipfile.ZipFile(output) as z:assert z.testzip() is None
print('Updated all five local projects. Verified ZIP:',output,round(output.stat().st_size/1024/1024,1),'MB')
