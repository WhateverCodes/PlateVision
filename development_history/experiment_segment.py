from pathlib import Path
p=Path('outputs/PlateVision1/src/character_segmenter.py');s=p.read_text();start=s.index('    for adaptive in (False, True):');end=s.index('        for mask in (binary, 255-binary):',start)
s=s[:start]+'''    otsu,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    binaries=[binary,cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,9)]
    for block in (21,41,61):
        for offset in (5,13):
            binaries.append(cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,block,offset))
    for offset in (-25,25):binaries.append(cv2.threshold(gray,max(1,min(254,otsu+offset)),255,cv2.THRESH_BINARY)[1])
    for binary in binaries:
'''+s[end:]
p.write_text(s)
