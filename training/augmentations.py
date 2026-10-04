"""Mild label-preserving photometric and geometric augmentation."""
import cv2
import numpy as np

def augment(image: np.ndarray, boxes: np.ndarray | None=None):
    h,w=image.shape[:2]
    angle=np.random.uniform(-5,5)
    matrix=cv2.getRotationMatrix2D((w/2,h/2),angle,np.random.uniform(0.95,1.05))
    matrix[:,2]+=np.random.uniform(-0.02,0.02,2)*[w,h]
    image=cv2.warpAffine(image,matrix,(w,h),borderValue=114 if boxes is not None else 0)
    if boxes is not None and len(boxes):
        corners=np.stack((boxes[:,[0,1]],boxes[:,[2,1]],boxes[:,[2,3]],boxes[:,[0,3]]),axis=1)
        transformed=np.concatenate((corners,np.ones((*corners.shape[:2],1))),axis=2)@matrix.T
        boxes=np.concatenate((transformed.min(1),transformed.max(1)),axis=1)
        boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],0,w)
        boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],0,h)
    image=np.clip(image.astype(float)*np.random.uniform(0.8,1.2)+np.random.uniform(-15,15),0,255).astype(np.uint8)
    if np.random.rand()<0.3:
        image=cv2.GaussianBlur(image,(3,3),0.6)
    if np.random.rand()<0.3:
        image=np.clip(image.astype(float)+np.random.normal(0,3,image.shape),0,255).astype(np.uint8)
    if np.random.rand()<0.2:
        _,encoded=cv2.imencode(".jpg",image,[cv2.IMWRITE_JPEG_QUALITY,int(np.random.randint(65,96))])
        image=cv2.imdecode(encoded,cv2.IMREAD_GRAYSCALE)
    return image,boxes
