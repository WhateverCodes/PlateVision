"""Bounded streaming video processing, with a full validation pass before inference."""
from dataclasses import dataclass
from pathlib import Path
import math
import cv2
from .timestamp_manager import Registry
from .visualization import annotate,jpeg

@dataclass
class VideoInfo:
    fps: float
    frames: int
    seconds: float

def inspect_video(path: Path,settings) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("Video could not be opened. Try converting it to MP4/H.264.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no usable frame rate; convert it to constant-frame-rate MP4.")
        count = 0
        last_seconds = 0.0
        while True:
            ok,frame = cap.read()
            if not ok:
                break
            count += 1
            reported = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if math.isfinite(reported) and reported >= 0:
                last_seconds = max(last_seconds, reported)
                if reported > settings.max_video_seconds:
                    raise ValueError("Video timestamps exceed the duration limit. It was not trimmed.")
            if frame.shape[0]*frame.shape[1] > settings.max_video_pixels:
                raise ValueError("Video resolution is too large. Use 1080p or smaller.")
            if count > settings.max_decode_frames:
                raise ValueError("Video exceeds the safe frame-count limit.")
            if count/fps > settings.max_video_seconds + 1e-6:
                raise ValueError(f"Video is longer than {settings.max_video_seconds:g} seconds. It was not trimmed.")
        if not count:
            raise ValueError("Video contains no decodable frames.")
        advertised = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if advertised > count+1:
            raise ValueError("Video decoding ended early; the file may be corrupted.")
        return VideoInfo(fps,count,max(count/fps,last_seconds+1/fps))
    finally:
        cap.release()

def process_video(path: Path,pipeline,settings,progress=None) -> tuple[Registry,VideoInfo]:
    info = inspect_video(path,settings)
    step = max(settings.frame_skip,math.ceil(info.fps/settings.target_fps),1)
    registry = Registry(settings.fuzzy_threshold)
    cap = cv2.VideoCapture(str(path))
    try:
        for index in range(info.frames):
            ok,frame = cap.read()
            if not ok:
                raise ValueError("Video decoding failed during processing.")
            if index%step == 0:
                # POS_MSEC preserves decoder timestamps when available (including VFR files).
                reported = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                seconds = reported if math.isfinite(reported) and reported > 0 else index/info.fps
                if seconds > settings.max_video_seconds:
                    raise ValueError("Decoded timestamps exceed the video duration limit.")
                detections = pipeline(frame)
                event = registry.observe(detections,seconds,index,info.fps)
                if event:
                    if len(registry.events)>settings.max_events:
                        raise ValueError("Too many events for this session. Use a shorter clip.")
                    event.image = jpeg(annotate(frame,detections))
            if progress:
                progress((index+1)/info.frames)
    finally:
        cap.release()
    return registry,info
