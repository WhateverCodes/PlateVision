"""Plate -> detected text lines -> CTC strings; never segments individual glyphs."""
from pathlib import Path
import cv2
import numpy as np
from .pretrained import session
from .paddle_candidate import PaddleCandidate


def clean_line(text,scores):
    letters=[];kept=[]
    for letter,score in zip(text.upper(),scores):
        if letter in ' .-\t\r\n':continue
        letters.append(letter);kept.append(score)
    return ''.join(letters),kept


def order_lines(items):
    if not items:return []
    height=float(np.median([v['height'] for v in items]))
    slopes=[(b['cy']-a['cy'])/(b['cx']-a['cx']) for i,a in enumerate(items) for b in items[i+1:]
            if abs(b['cx']-a['cx'])>1.5*height
            and abs((b['cy']-a['cy'])/(b['cx']-a['cx']))<.6]
    slope=float(np.median(slopes)) if slopes else 0.
    def level(item):return item['cy']-slope*item['cx']
    rows=[]
    for item in sorted(items,key=level):
        row=next((r for r in rows if abs(level(item)-np.mean([level(v) for v in r]))<.5*min(item['height'],np.median([v['height'] for v in r]))),None)
        if row is None:rows.append([item])
        else:row.append(item)
    return [v for r in rows for v in sorted(r,key=lambda v:v['cx'])]


class WholeLineReader:
    def __init__(self,folder):
        folder=Path(folder)
        self.reader=PaddleCandidate(folder/'english.onnx')
        self.detector=session(folder/'text_detector.onnx',2)
        self.input=self.detector.get_inputs()[0].name
        self.provenance='Pretrained PaddleOCR PP-OCRv5 mobile text-line detection and English CTC recognition, RapidAI ONNX exports. Local CPU, two threads. No character cutting or own CNN weights.'
        if (folder/'adaptation.json').exists():
            self.provenance+=' Final CTC projection adapted locally on 76 reviewed training lines, with a conservative 50% parameter update. Feature extractor remains pretrained; development-selected, not independently validated.'

    def line_crops(self,image):
        h,w=image.shape[:2]
        scale=640/max(h,w)
        rh=max(32,round(h*scale/32)*32);rw=max(32,round(w*scale/32)*32)
        resized=cv2.resize(image,(rw,rh)).astype(np.float32)/255
        normalized=(resized-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        inputs=np.ascontiguousarray(normalized.transpose(2,0,1)[None])
        prob=self.detector.run(None,{self.input:inputs})[0].squeeze()
        mask=(prob>.3).astype(np.uint8)*255
        contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        items=[]
        for contour in sorted(contours,key=cv2.contourArea,reverse=True)[:30]:
            if cv2.contourArea(contour)<12:continue
            region=np.zeros(prob.shape,np.uint8);cv2.drawContours(region,[contour],-1,1,-1)
            if cv2.mean(prob,mask=region)[0]<.5:continue
            center,size,angle=cv2.minAreaRect(contour)
            if min(size)<3:continue
            expansion=cv2.contourArea(contour)*1.5/max(cv2.arcLength(contour,True),1)
            points=cv2.boxPoints((center,(size[0]+2*expansion,size[1]+2*expansion),angle))
            points[:,0]*=w/prob.shape[1];points[:,1]*=h/prob.shape[0]
            sums=points.sum(1);diff=np.diff(points,axis=1).ravel()
            corners=np.array([points[sums.argmin()],points[diff.argmin()],points[sums.argmax()],points[diff.argmax()]],np.float32)
            if len(np.unique(corners,axis=0))!=4:continue
            cw=round(max(np.linalg.norm(corners[1]-corners[0]),np.linalg.norm(corners[2]-corners[3])))
            ch=round(max(np.linalg.norm(corners[3]-corners[0]),np.linalg.norm(corners[2]-corners[1])))
            if cw<8 or ch<4:continue
            target=np.float32([[0,0],[cw-1,0],[cw-1,ch-1],[0,ch-1]])
            crop=cv2.warpPerspective(image,cv2.getPerspectiveTransform(corners,target),(cw,ch),borderMode=cv2.BORDER_REPLICATE)
            items.append({'crop':crop,'cx':float(points[:,0].mean()),'cy':float(points[:,1].mean()),'height':ch})
        return order_lines(items[:8])

    def read(self,image):
        text,scores=self._read_once(image)
        # Retry only an uncertain reading. A larger view may recover small
        # strokes, but disagreement must never overwrite the original text.
        if scores and np.mean(scores)<.95 and max(image.shape[:2])<=640:
            enlarged=cv2.resize(image,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
            other,other_scores=self._read_once(enlarged)
            if other==text and other_scores and np.mean(other_scores)>np.mean(scores):
                return text,other_scores
        return text,scores

    def _read_once(self,image):
        from .validator import plausible_ocr as plausible
        if image.ndim==2:image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        lines=self.line_crops(image)
        if not lines:lines=[{'crop':image}]
        beams=[('',[],0.)]
        for line in lines:
            alternatives=[(*clean_line(text,scores),sum(float(np.log(max(s,1e-8))) for s in scores))
                          for text,scores in self.reader.alternatives(line['crop'])]
            if alternatives and alternatives[0][0]=='IND':continue
            beams=sorted([(text+reading,scores+confidence,quality+line_quality)
                for text,scores,quality in beams for reading,confidence,line_quality in alternatives],key=lambda v:v[2],reverse=True)[:24]
        if not beams:return '',[]
        # Preserve an already plausible top prediction. Only attempt a format
        # repair when the OCR's top string fails the existing registration rule.
        eligible=[v for v in beams if plausible(v[0])]
        selected=beams[0] if plausible(beams[0][0]) or not eligible else eligible[0]
        return selected[0],selected[1]
