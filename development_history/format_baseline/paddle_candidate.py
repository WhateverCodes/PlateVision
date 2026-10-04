"""Local PP-OCRv5 English reader; no external service calls.

ONNX export: RapidAI/RapidOCR v3.9.2. Input normalization and CTC dictionary
conventions follow RapidOCR. Used by the application's whole-line reader.
"""
import math
import cv2
import numpy as np
from .pretrained import session


class PaddleCandidate:
    def __init__(self,path):
        self.model=session(path,2)
        self.input=self.model.get_inputs()[0].name
        self.alphabet=['']+self.model.get_modelmeta().custom_metadata_map['character'].splitlines()+[' ']
        self.provenance='Pretrained PP-OCRv5 English mobile, RapidAI ONNX export; local CPU; external weights.'

    def predict_tokens(self,image):
        if image.ndim==2:image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        h,w=image.shape[:2]
        width=min(1280,max(320,math.ceil(48*w/h)))
        resized_width=min(width,math.ceil(48*w/h))
        resized=cv2.resize(image,(resized_width,48)).astype(np.float32).transpose(2,0,1)/127.5-1
        batch=np.zeros((1,3,48,width),np.float32);batch[0,:,:,:resized_width]=resized
        values=self.model.run(None,{self.input:batch})[0][0]
        if values.shape[-1]!=len(self.alphabet):raise ValueError('OCR dictionary/model mismatch')
        return values

    def read(self,image):
        values=self.predict_tokens(image)
        ids=values.argmax(-1);scores=values.max(-1)
        text=[];confidence=[];last=0
        for token,score in zip(ids,scores):
            if token and token!=last:
                text.append(self.alphabet[token]);confidence.append(float(score))
            last=token
        return ''.join(text),confidence

    def alternatives(self,image,beam_width=12):
        """Alternatives at emitted CTC positions, using unmodified model scores.

        This preserves greedy token boundaries; it is not a full CTC beam search.
        No insertions, deletions or hard-coded letter/digit substitutions occur.
        """
        values=self.predict_tokens(image);ids=values.argmax(-1)
        beams=[('',[],0.)];last=0
        for token,distribution in zip(ids,values):
            if token and token!=last:
                options=[]
                for index in distribution.argsort()[-3:][::-1]:
                    if index==0:continue
                    score=float(distribution[index])
                    if index!=token and (score<.02 or score<float(distribution[token])*.05):continue
                    options.append((self.alphabet[index],score))
                beams=sorted([(text+letter,scores+[score],quality+float(np.log(max(score,1e-8))))
                    for text,scores,quality in beams for letter,score in options],key=lambda v:v[2],reverse=True)[:beam_width]
            last=token
        return [(text,scores) for text,scores,_ in beams]
