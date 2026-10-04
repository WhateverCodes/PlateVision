from pathlib import Path
p=Path('outputs/PlateVision1/src/character_segmenter.py');s=p.read_text();s=s.replace('    return candidates\n','''    # Normalize slanted glyph strokes as an alternative; preserve originals.
    alternatives=[]
    for score,chars,mask in candidates:
        corrected=[]
        for char in chars:
            moments=cv2.moments(char.image)
            skew=moments['mu11']/moments['mu02'] if abs(moments['mu02'])>1e-5 else 0.
            skew=float(np.clip(skew,-.5,.5))
            matrix=np.float32([[1,skew,-16*skew],[0,1,0]])
            glyph=cv2.warpAffine(char.image,matrix,(32,32),flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)
            corrected.append(Character(char.box,glyph))
        alternatives.append((score,corrected,mask))
    return candidates+alternatives
''');p.write_text(s)
