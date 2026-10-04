"""Adapt only PaddleOCR's final CTC projection from its existing ONNX export."""
import sys,json,copy,math,time,hashlib
from pathlib import Path
root=Path.cwd();sys.path.insert(0,str(root/'work/onnx_tools'));sys.path.insert(0,str(root/'outputs/PlateVision1'))
import cv2,numpy as np,torch,onnx
from onnx import numpy_helper,helper,TensorProto
from src.runtime import configure
from src.pretrained import session
configure(2)
p=root/'outputs/PlateVision1';data=p/'data/ocr_finetune_audit_v1';out=p/'models/paddle_head_trial_v1'
if out.exists():raise RuntimeError('Preserve prior trial; use a new directory')
out.mkdir(parents=True)
source=p/'models/paddle_candidate/english.onnx';model=onnx.load(str(source));initial={v.name:numpy_helper.to_array(v).copy() for v in model.graph.initializer}
matmul=model.graph.node[-4];assert matmul.op_type=='MatMul'
feature_name,weight_name=matmul.input;add=model.graph.node[-3];assert add.op_type=='Add';bias_name=add.input[1]
w0=initial[weight_name];b0=initial[bias_name];alphabet=['']+next(v.value for v in model.metadata_props if v.key=='character').splitlines()+[' '];assert len(alphabet)==len(b0)
features_model=copy.deepcopy(model);features_model.graph.output.append(helper.make_tensor_value_info(feature_name,TensorProto.FLOAT,[None,None,w0.shape[0]]));onnx.save(features_model,str(out/'features.onnx'));reader=session(out/'features.onnx',2);input_name=reader.get_inputs()[0].name

def encode(image):
 h,w=image.shape[:2];width=min(1280,max(320,math.ceil(48*w/h)));rw=min(width,math.ceil(48*w/h));im=cv2.resize(image,(rw,48)).astype(np.float32).transpose(2,0,1)/127.5-1;batch=np.zeros((1,3,48,width),np.float32);batch[0,:,:,:rw]=im;return batch

def cache(rows):
 result=[]
 for r in rows:
  prob,features=reader.run(None,{input_name:encode(cv2.imread(r['image']))})
  predicted=torch.softmax(torch.from_numpy(features)@torch.from_numpy(w0)+torch.from_numpy(b0),-1).numpy()
  if not np.allclose(prob,predicted,atol=2e-5,rtol=2e-4):raise ValueError('Extracted head does not reproduce model')
  result.append((torch.from_numpy(features[0]),torch.tensor([alphabet.index(c) for c in r['text']],dtype=torch.long),r['text']))
 return result
train_rows=json.loads((data/'train.json').read_text());val_rows=json.loads((data/'validation_candidates.json').read_text())
assert not ({r['id'] for r in train_rows}&{r['id'] for r in val_rows});assert not ({r['text'] for r in train_rows}&{r['text'] for r in val_rows})
train=cache(train_rows);val=cache(val_rows);print('Cached train/validation features:',len(train),len(val),'Head reproduction verified',flush=True)
w=torch.nn.Parameter(torch.from_numpy(w0));b=torch.nn.Parameter(torch.from_numpy(b0));optimizer=torch.optim.AdamW([w,b],lr=1e-4,weight_decay=1e-4);criterion=torch.nn.CTCLoss(blank=0,zero_infinity=False)
def evaluate():
 exact=0;results=[]
 with torch.no_grad():
  for features,labels,text in val:
   indices=(features@w+b).argmax(-1).tolist();last=0;chars=[]
   for i in indices:
    if i and i!=last:chars.append(alphabet[i])
    last=i
   prediction=''.join(chars).upper().replace(' ','').replace('.','').replace('-','');exact+=prediction==text;results.append({'expected':text,'predicted':prediction})
 return {'exact':exact,'images':len(val),'results':results}
baseline=evaluate();best=baseline['exact'];best_state=None;history=[];start=time.monotonic();rng=np.random.default_rng(42)
for epoch in range(1,9):
 losses=[]
 for offset in range(0,len(train),4):
  if time.monotonic()-start>120:raise RuntimeError('Training time limit reached')
  if offset==0:order=rng.permutation(len(train))
  batch=[train[i] for i in order[offset:offset+4]];lengths=torch.tensor([len(r[0]) for r in batch]);target_lengths=torch.tensor([len(r[1]) for r in batch]);features=torch.nn.utils.rnn.pad_sequence([r[0] for r in batch]);targets=torch.cat([r[1] for r in batch]);logprobs=(features@w+b).log_softmax(-1)
  optimizer.zero_grad();loss=criterion(logprobs,targets,lengths,target_lengths)
  if not torch.isfinite(loss):raise ValueError('Nonfinite CTC loss')
  loss.backward();torch.nn.utils.clip_grad_norm_([w,b],1);optimizer.step();losses.append(float(loss.detach()))
 metric=evaluate();history.append({'epoch':epoch,'loss':float(np.mean(losses)),**metric});print('epoch',epoch,'val exact',metric['exact'],'/',len(val),flush=True)
 if metric['exact']>best:best=metric['exact'];best_state=(w.detach().numpy().copy(),b.detach().numpy().copy(),epoch)
report={'method':'Final CTC linear layer adaptation only; frozen pretrained features, existing ONNX weights, PyTorch CTC loss. Not full PaddleOCR fine-tuning.','training_images':len(train),'validation_images':len(val),'baseline':baseline,'history':history,'seconds':time.monotonic()-start,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'selected_epoch':best_state[2] if best_state else None,'note':'Validation subset is development data, not an independent test. Full 45-crop pipeline comparison required before promotion.'}
if best_state:
 for i,tensor in enumerate(model.graph.initializer):
  if tensor.name in [weight_name,bias_name]:model.graph.initializer[i].CopyFrom(numpy_helper.from_array(best_state[0] if tensor.name==weight_name else best_state[1],tensor.name))
 onnx.checker.check_model(model);onnx.save(model,str(out/'english.onnx'))
(out/'report.json').write_text(json.dumps(report,indent=2));print('Baseline',baseline['exact'],'Best',best,'Candidate saved',bool(best_state),flush=True)
