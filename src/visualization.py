"""Serial-only box labels and compact event images."""
import cv2
import numpy as np

PALETTE = [(220,240,20),(90,255,150),(255,100,195),(40,190,255)]

def annotate(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    out = image.copy()
    for d in detections:
        if d.get("recognized") is False:
            continue
        serial = d.get("serial")
        color = PALETTE[(int(serial)-1)%len(PALETTE)] if serial else (140,140,140)
        box = d.get("vehicle_box") or d["box"]
        x1,y1,x2,y2 = map(int,box)
        cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
        if serial:
            cv2.putText(out,serial,(x1+4,max(y1+22,y2-8)),cv2.FONT_HERSHEY_SIMPLEX,0.65,color,2,cv2.LINE_AA)
        if d.get("vehicle_box"):
            a,b,c,e = map(int,d["box"])
            cv2.rectangle(out,(a,b),(c,e),color,1)
    return out

def jpeg(image: np.ndarray, max_width: int = 1280) -> bytes:
    if image.shape[1] > max_width:
        image = cv2.resize(image,(max_width,round(image.shape[0]*max_width/image.shape[1])))
    ok,buffer = cv2.imencode(".jpg",image,[cv2.IMWRITE_JPEG_QUALITY,85])
    if not ok:
        raise ValueError("Could not encode the event image.")
    return buffer.tobytes()
