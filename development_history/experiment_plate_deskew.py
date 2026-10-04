from pathlib import Path
p=Path('outputs/PlateVision1/src/character_segmenter.py');s=p.read_text();start=s.index('    # Normalize slanted glyph');end=s.index('\ndef segment',start);s=s[:start]+'    return candidates\n'+s[end:];p.write_text(s)
p=Path('outputs/PlateVision1/src/character_classifier.py');s=p.read_text();s=s.replace('        candidates=segmentation_candidates(plate)','''        candidates=segmentation_candidates(plate)
        # Estimate a shared text baseline from character centres, then try a
        # whole-crop rotation while retaining the original candidates.
        import cv2
        reference=max(candidates,key=lambda v:v[0])[1]
        if len(reference)>=4:
            height=float(np.median([c.box[3]-c.box[1] for c in reference]))
            centres=[((c.box[0]+c.box[2])/2,(c.box[1]+c.box[3])/2) for c in reference]
            slopes=[(b[1]-a[1])/(b[0]-a[0]) for i,a in enumerate(centres) for b in centres[i+1:]
                    if abs(b[0]-a[0])>2*height and abs((b[1]-a[1])/(b[0]-a[0]))<.36]
            angle=float(np.degrees(np.arctan(np.median(slopes)))) if slopes else 0.
            if 2<abs(angle)<20:
                h,w=plate.shape[:2];margin=max(4,int(max(h,w)*.15))
                padded=cv2.copyMakeBorder(plate,margin,margin,margin,margin,cv2.BORDER_REPLICATE)
                ph,pw=padded.shape[:2];matrix=cv2.getRotationMatrix2D((pw/2,ph/2),angle,1)
                aligned=cv2.warpAffine(padded,matrix,(pw,ph),borderMode=cv2.BORDER_REPLICATE)
                candidates+=segmentation_candidates(aligned)''');p.write_text(s)

