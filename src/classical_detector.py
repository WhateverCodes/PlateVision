"""Bounded image-processing plate proposals; scores are geometric heuristics."""
import cv2
import numpy as np
from .preprocessing import grayscale

class ClassicalDetector:
    provenance="Image-processing proposals: morphology and rectangular contours; no learned detection weights. Scores are heuristic."
    def __call__(self,image):
        gray=grayscale(image);h,w=gray.shape
        scale=min(1.,1000/max(h,w));small=cv2.resize(gray,(max(1,round(w*scale)),max(1,round(h*scale))))
        candidates=[]
        for kernel in [(17,5),(29,9),(41,13)]:
            for op in (cv2.MORPH_BLACKHAT,cv2.MORPH_TOPHAT):
                response=cv2.morphologyEx(small,op,cv2.getStructuringElement(cv2.MORPH_RECT,kernel))
                binary=cv2.threshold(response,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
                joined=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(kernel[0],3)))
                contours,_=cv2.findContours(joined,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x,y,cw,ch=cv2.boundingRect(contour);aspect=cw/max(ch,1)
                    if not (1.4<aspect<8 and cw>=35 and ch>=9 and cw*ch<small.size*.4):continue
                    fill=cv2.contourArea(contour)/(cw*ch)
                    if fill<.25:continue
                    pad_x=max(2,round(cw*.05));pad_y=max(2,round(ch*.2))
                    box=[max(0,int((x-pad_x)/scale)),max(0,int((y-pad_y)/scale)),min(w,int((x+cw+pad_x)/scale)),min(h,int((y+ch+pad_y)/scale))]
                    score=.5*min(fill,1)+.5*max(0,1-abs(aspect-4)/6)
                    candidates.append({"box":box,"score":score})
        # Plate backgrounds can be a connected light/dark region even when text
        # morphology misses them. Keep these as additional geometric proposals.
        threshold=cv2.threshold(small,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        for mask in (threshold,255-threshold):
            contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x,y,cw,ch=cv2.boundingRect(contour);aspect=cw/max(ch,1)
                if not (1.1<aspect<8 and cw>=35 and ch>=12 and cw*ch<small.size*.4):continue
                fill=cv2.contourArea(contour)/(cw*ch)
                if fill<.55:continue
                candidates.append({"box":[int(x/scale),int(y/scale),min(w,int((x+cw)/scale)),min(h,int((y+ch)/scale))],"score":.5*min(fill,1)+.5*max(0,1-abs(aspect-4)/6)})
        for candidate in candidates:
            a,b,c,d=candidate["box"]
            roi=gray[b:d,a:c]
            if not roi.size:continue
            roi=cv2.resize(roi,(max(16,round(roi.shape[1]*64/roi.shape[0])),64))
            binary=cv2.threshold(roi,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            best=0
            for mask in (binary,255-binary):
                _,_,stats,_=cv2.connectedComponentsWithStats(mask,8)
                n=sum(18<=hh<=59 and 2<=ww<=hh*.95 and xx>0 and yy>0 and area/(ww*hh)>.15 for xx,yy,ww,hh,area in stats[1:])
                if 5<=n<=12:best=max(best,1-abs(n-9)/9)
            candidate["score"]=.4*candidate["score"]+.6*best
        kept=[]
        for r in sorted(candidates,key=lambda r:r["score"],reverse=True):
            a=r["box"]
            def overlap(b):
                intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
                return intersection/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)
            if all(overlap(v["box"])<.35 for v in kept):kept.append(r)
            if len(kept)==15:break
        return kept

class PlateCropDetector:
    provenance="User selected an already-cropped plate; automatic localization bypassed."
    def __call__(self,image):
        h,w=image.shape[:2]
        return [{"box":[0,0,w,h],"score":1.}]
