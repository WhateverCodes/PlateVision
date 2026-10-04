import sys
sys.path.insert(0,'work/onnx_tools')
import onnx
m=onnx.load('outputs/PlateVision1/models/paddle_candidate/english.onnx')
for n in m.graph.node[-10:]:print(n.op_type,list(n.input),list(n.output))
print('outputs',m.graph.output)
