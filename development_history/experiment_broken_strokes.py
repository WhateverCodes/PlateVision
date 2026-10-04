from pathlib import Path
p=Path('outputs/PlateVision1/src/character_classifier.py');s=p.read_text();start=s.index('        # Estimate a shared');end=s.index('        baseline=',start);s=s[:start]+s[end:];p.write_text(s)
p=Path('outputs/PlateVision1/src/character_segmenter.py');s=p.read_text().replace('        for mask in (binary, 255-binary):','''        masks=[binary,255-binary]
        masks += [cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,1),np.uint8)) for mask in masks]
        for mask in masks:''');p.write_text(s)
