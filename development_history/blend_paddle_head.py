import sys,copy
from pathlib import Path
sys.path.insert(0,'work/onnx_tools')
import onnx
from onnx import numpy_helper
p=Path('outputs/PlateVision1/models');base=onnx.load(str(p/'paddle_candidate/english.onnx'));candidate=onnx.load(str(p/'paddle_head_trial_v1/english.onnx'));original={v.name:numpy_helper.to_array(v) for v in base.graph.initializer}
for i,t in enumerate(candidate.graph.initializer):
 if t.name in ['linear_8.w_0','linear_8.b_0']:
  value=(original[t.name]+numpy_helper.to_array(t))*.5;candidate.graph.initializer[i].CopyFrom(numpy_helper.from_array(value,t.name))
out=p/'paddle_head_half_trial_v1';out.mkdir(exist_ok=True);onnx.checker.check_model(candidate);onnx.save(candidate,str(out/'english.onnx'))
s=Path('work/evaluate_paddle_head.py').read_text(encoding='utf-8-sig').replace('paddle_head_trial_v1','paddle_head_half_trial_v1');Path('work/evaluate_paddle_half.py').write_text(s,encoding='utf8')
