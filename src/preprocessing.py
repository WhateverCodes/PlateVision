"""Shared training/inference preprocessing; coordinates remain traceable."""
from io import BytesIO
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps

def read_image(source: bytes | Path, max_pixels: int = 24_000_000) -> np.ndarray:
    try:
        with Image.open(BytesIO(source) if isinstance(source, bytes) else source) as im:
            if im.width * im.height > max_pixels:
                raise ValueError("Image is too large. Resize it before uploading.")
            rgb = np.array(ImageOps.exif_transpose(im).convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The image could not be read. Try a valid JPG or PNG.") from exc

def grayscale(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Image is empty.")
    return image.copy() if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def enhance(gray: np.ndarray, clahe: bool = False, bilateral: bool = False,
            sharpen: bool = False) -> np.ndarray:
    out = gray.copy()
    if clahe:
        out = cv2.createCLAHE(2.0, (8, 8)).apply(out)
    if bilateral:
        out = cv2.bilateralFilter(out, 5, 30, 30)
    if sharpen:
        out = cv2.addWeighted(out, 1.5, cv2.GaussianBlur(out, (0, 0), 1), -0.5, 0)
    return out

def letterbox(gray: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    if size < 64 or size % 32:
        raise ValueError("Image size must be a multiple of 32 and at least 64.")
    h, w = gray.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    x, y = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size), 114, np.uint8)
    canvas[y:y+nh, x:x+nw] = cv2.resize(gray, (nw, nh))
    return canvas, scale, (x, y)

def normalize_character(mask: np.ndarray, size: int = 32) -> np.ndarray:
    """White foreground on black, tight crop, aspect-preserving centered padding."""
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return np.zeros((size, size), np.uint8)
    crop = mask[ys.min():ys.max()+1, xs.min():xs.max()+1]
    h, w = crop.shape
    scale = (size - 8) / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    out = np.zeros((size, size), np.uint8)
    x, y = (size - nw)//2, (size - nh)//2
    out[y:y+nh, x:x+nw] = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    return out

def rectify_plate(image: np.ndarray) -> np.ndarray:
    """Only warp a large convex quadrilateral; otherwise preserve the crop."""
    gray = grayscale(image)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    area = gray.size
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        poly = cv2.approxPolyDP(contour, 0.025*cv2.arcLength(contour, True), True)
        if len(poly) != 4 or not cv2.isContourConvex(poly) or cv2.contourArea(poly) < area*0.65:
            continue
        p = poly[:, 0].astype(np.float32)
        sums, diffs = p.sum(1), np.diff(p, axis=1).ravel()
        q = np.array([p[sums.argmin()], p[diffs.argmin()], p[sums.argmax()], p[diffs.argmax()]])
        if len(np.unique(q, axis=0)) != 4:
            continue
        w = round(max(np.linalg.norm(q[1]-q[0]), np.linalg.norm(q[2]-q[3])))
        h = round(max(np.linalg.norm(q[3]-q[0]), np.linalg.norm(q[2]-q[1])))
        if w < 30 or h < 12:
            continue
        dst = np.float32([[0,0], [w-1,0], [w-1,h-1], [0,h-1]])
        return cv2.warpPerspective(image, cv2.getPerspectiveTransform(q, dst), (w,h))
    return image
