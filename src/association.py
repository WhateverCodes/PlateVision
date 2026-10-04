"""Geometric association, with Unknown when there is no credible vehicle match."""
import numpy as np

def associate(plate: list[int], vehicles: list[dict]) -> dict | None:
    x1,y1,x2,y2 = plate
    area = max(1,(x2-x1)*(y2-y1))
    matches = []
    for vehicle in vehicles:
        a,b,c,d = vehicle["box"]
        overlap = max(0,min(x2,c)-max(x1,a))*max(0,min(y2,d)-max(y1,b))/area
        if overlap >= 0.8:
            matches.append((max(1,(c-a)*(d-b)),vehicle))
    if matches:
        return min(matches,key=lambda x:x[0])[1]
    # Allow a small boundary error only; do not assign a distant vehicle.
    cx,cy = (x1+x2)/2,(y1+y2)/2
    close = []
    for v in vehicles:
        a,b,c,d = v["box"]
        distance = np.hypot(max(a-cx,0,cx-c),max(b-cy,0,cy-d))/max(1,np.hypot(c-a,d-b))
        if distance < 0.03:
            close.append((distance,v))
    return min(close,key=lambda x:x[0])[1] if close else None
