from pathlib import Path
import zipfile
p=Path('outputs/PlateVision1/training/adapt_paddle_projection.py');s=p.read_text();s=s.replace('w=torch.nn.Parameter(torch.from_numpy(w0));b=torch.nn.Parameter(torch.from_numpy(b0))','w=torch.nn.Parameter(torch.from_numpy(w0.copy()));b=torch.nn.Parameter(torch.from_numpy(b0.copy()))');p.write_text(s)
import shutil
for t in Path('outputs').glob('PlateVision*'):
 if t.name!='PlateVision1' and (t/'app.py').exists():shutil.copy2(p,t/'training/adapt_paddle_projection.py')
zpath=Path('outputs/PLATEVISION_Paddle_Update_2026-10-03.zip');temp=zpath.with_suffix('.tmp')
with zipfile.ZipFile(zpath) as old,zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as new:
 for item in old.infolist():new.writestr(item,p.read_bytes() if item.filename.endswith('/training/adapt_paddle_projection.py') else old.read(item.filename))
with zipfile.ZipFile(temp) as z:assert z.testzip() is None
temp.replace(zpath)
import ast;ast.parse(s);print('Training script checked; ZIP verified.')
