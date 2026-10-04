"""Optional slant normalization based solely on glyph pixels."""
import cv2
import numpy as np
from .character_segmenter import Character


def deskew_characters(chars):
    if not chars or not all(isinstance(c,Character) for c in chars):return None
    result=[];changed=False
    for c in chars:
        moments=cv2.moments(c.image)
        if moments['mu02']<1e-6:
            result.append(c);continue
        skew=moments['mu11']/moments['mu02']
        if not .08<=abs(skew)<=.7:
            result.append(c);continue
        h,w=c.image.shape
        transform=np.float32([[1,skew,-.5*w*skew],[0,1,0]])
        im=cv2.warpAffine(c.image,transform,(w,h),flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)
        result.append(Character(c.box,im));changed=True
    return result if changed else None
