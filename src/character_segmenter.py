"""Contour-based segmentation, independent of OCR libraries."""
from dataclasses import dataclass
import cv2
import numpy as np
from .preprocessing import grayscale, normalize_character

@dataclass
class Character:
    box: tuple[int, int, int, int]
    image: np.ndarray

def order_characters(chars: list[Character]) -> list[Character]:
    if not chars:
        return []
    # Estimate plate tilt from well-separated letters. Clustering raw y centres
    # can split a sloping single line and read its right-hand end first.
    height = float(np.median([c.box[3]-c.box[1] for c in chars]))
    centres = [((c.box[0]+c.box[2])/2, (c.box[1]+c.box[3])/2) for c in chars]
    slopes = [(y2-y1)/(x2-x1) for i,(x1,y1) in enumerate(centres)
              for x2,y2 in centres[i+1:] if abs(x2-x1)>2*height
              and abs((y2-y1)/(x2-x1))<0.6]
    slope = float(np.median(slopes)) if slopes else 0.0
    def level(c):
        return (c.box[1]+c.box[3])/2-slope*(c.box[0]+c.box[2])/2
    rows: list[list[Character]] = []
    for c in sorted(chars, key=level):
        cy = level(c)
        row = next((r for r in rows if abs(cy - np.mean([level(v) for v in r]))
                    < 0.5*np.median([v.box[3]-v.box[1] for v in r])), None)
        if row is None:
            rows.append([c])
        else:
            row.append(c)
    return [c for row in rows for c in sorted(row, key=lambda c: c.box[0])]

def segmentation_candidates(image: np.ndarray):
    gray = grayscale(image)
    if min(gray.shape) < 8:
        return [(0, [], gray)]
    scale = 128 / gray.shape[0]
    gray = cv2.resize(gray, (max(16, round(gray.shape[1]*scale)), 128))
    candidates = []
    for adaptive in (False, True):
        binary = (cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                  cv2.THRESH_BINARY, 31, 9) if adaptive else
                  cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1])
        for mask in (binary, 255-binary):
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            chars = []
            for x,y,w,h,area in stats[1:]:
                if not (25 <= h <= 118 and 2 <= w <= h*1.05 and 0.07 <= w/h <= 1.05):
                    continue
                if area/(w*h) < 0.12 or x <= 0 or y <= 0 or x+w >= gray.shape[1]:
                    continue
                patch = mask[y:y+h,x:x+w]
                chars.append(Character((int(x),int(y),int(x+w),int(y+h)), normalize_character(patch)))
            ordered = order_characters(chars)
            score = len(ordered) if 6 <= len(ordered) <= 12 else -abs(len(ordered)-9)
            candidates.append((score, ordered, mask))
    return candidates

def segment(image: np.ndarray) -> tuple[list[Character], np.ndarray]:
    _, chars, mask = max(segmentation_candidates(image), key=lambda v: v[0])
    return chars, mask
