"""Local CPU adapters for ankandrew's pretrained ONNX plate models.

Input/output conventions follow fast-plate-ocr and open-image-models (MIT).
Their notices and source URLs are included in models/pretrained. No downloads
or network calls occur in these adapters.
"""
from pathlib import Path
import cv2
import numpy as np


def session(path, threads=2):
    import onnxruntime as ort
    if not Path(path).exists():
        raise FileNotFoundError('Pretrained model missing. Run SETUP_PRETRAINED.cmd once.')
    options=ort.SessionOptions()
    options.intra_op_num_threads=threads
    options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])


class PretrainedReader:
    def __init__(self,folder,threads=2):
        import yaml
        folder=Path(folder)
        self.config=yaml.safe_load((folder/'cct_s_v2_global_plate_config.yaml').read_text())
        self.model=session(folder/'cct_s_v2_global.onnx',threads)
        self.input=self.model.get_inputs()[0].name
        self.output='plate' if 'plate' in [v.name for v in self.model.get_outputs()] else self.model.get_outputs()[0].name
        self.provenance='Pretrained FastPlateOCR CCT-S v2 global (ankandrew), unchanged external weights; CPU ONNX. Not our trained model. Maximum 10 characters.'

    def read(self,image):
        cfg=self.config
        # The selected model takes RGB uint8 NHWC; normalization is in its graph.
        rgb=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB if image.ndim==2 else cv2.COLOR_BGR2RGB)
        rgb=cv2.resize(rgb,(cfg['img_width'],cfg['img_height']),interpolation=cv2.INTER_LINEAR)
        values=self.model.run([self.output],{self.input:rgb[None].astype(np.uint8)})[0]
        values=values.reshape(cfg['max_plate_slots'],len(cfg['alphabet']))
        indices=values.argmax(-1)
        text=''.join(cfg['alphabet'][i] for i in indices).rstrip(cfg['pad_char'])
        scores=values.max(-1)[:len(text)].astype(float).tolist()
        return text,scores


class PretrainedDetector:
    def __init__(self,folder,threshold=0.5,threads=2):
        self.model=session(Path(folder)/'yolo-v9-t-384-license-plates-end2end.onnx',threads)
        self.input=self.model.get_inputs()[0].name
        self.size=384
        self.threshold=threshold
        self.provenance='Pretrained YOLOv9-t 384 plate detector from ankandrew/open-image-models; unchanged external weights; CPU ONNX.'

    def __call__(self,image):
        detections=self._detect(image)
        h,w=image.shape[:2]
        if max(h,w)>640:
            # Four overlapping views retain small plates lost at 384px.
            tw,th=round(w*.6),round(h*.6)
            for x,y in ((0,0),(w-tw,0),(0,h-th),(w-tw,h-th)):
                for d in self._detect(image[y:y+th,x:x+tw]):
                    a,b,c,e=d['box'];d['box']=[a+x,b+y,c+x,e+y]
                    detections.append(d)
        kept=[]
        for d in sorted(detections,key=lambda v:v['score'],reverse=True):
            a,b,c,e=d['box'];area=(c-a)*(e-b)
            duplicate=False
            for old in kept:
                f,g,h,i=old['box'];intersection=max(0,min(c,h)-max(a,f))*max(0,min(e,i)-max(b,g))
                if intersection/max(1,area+(h-f)*(i-g)-intersection)>.4:
                    duplicate=True;break
            if not duplicate:kept.append(d)
        return kept[:30]

    def _detect(self,image):
        if image.ndim==2:image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        h,w=image.shape[:2];scale=min(self.size/w,self.size/h)
        nw,nh=round(w*scale),round(h*scale)
        dx,dy=(self.size-nw)/2,(self.size-nh)/2
        small=cv2.resize(image,(nw,nh))
        padded=cv2.copyMakeBorder(small,round(dy-.1),round(dy+.1),round(dx-.1),round(dx+.1),cv2.BORDER_CONSTANT,value=(114,114,114))
        tensor=np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255
        output=np.asarray(self.model.run(None,{self.input:tensor})[0]).reshape(-1,7)
        detections=[]
        for batch,x1,y1,x2,y2,label,score in output:
            if not np.isfinite([x1,y1,x2,y2,score]).all() or score<self.threshold:continue
            box=[max(0,min(w,int((x1-dx)/scale))),max(0,min(h,int((y1-dy)/scale))),
                 max(0,min(w,int((x2-dx)/scale))),max(0,min(h,int((y2-dy)/scale)))]
            if box[2]<=box[0] or box[3]<=box[1]:continue
            detections.append({'box':box,'score':float(score)})
        return sorted(detections,key=lambda d:d['score'],reverse=True)[:30]
