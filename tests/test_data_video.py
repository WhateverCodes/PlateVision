from dataclasses import replace
import cv2
import numpy as np
import pytest
from config import DEFAULTS
from training.data_tools import grouped_split
from src.video_processor import inspect_video,process_video

def test_no_group_hash_or_plate_leakage():
    rows=[{"group":str(i),"sha256":str(i),"texts":[f"PLATE{i}"]} for i in range(20)]
    rows[1]["sha256"]=rows[0]["sha256"]
    rows[2]["texts"]=rows[1]["texts"]
    splits=grouped_split(rows)
    memberships={id(r):k for k,rs in splits.items() for r in rs}
    assert memberships[id(rows[0])]==memberships[id(rows[1])]==memberships[id(rows[2])]
    assert splits==grouped_split(rows)

def make_video(path):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"MJPG"),10,(100,80))
    assert writer.isOpened()
    for i in range(20):
        writer.write(np.full((80,100,3),i,np.uint8))
    writer.release()

def test_overlong_video_is_rejected(tmp_path):
    path=tmp_path/"clip.avi"
    make_video(path)
    with pytest.raises(ValueError,match="longer"):
        inspect_video(path,replace(DEFAULTS,max_video_seconds=1))
    info=inspect_video(path,DEFAULTS)
    assert info.frames==20 and info.seconds==2

def test_streaming_event_frames_and_progress(tmp_path):
    path=tmp_path/"clip.avi"
    make_video(path)
    def pipeline(frame):
        return [{"text":"MH12AB1234","recognized":True,"confidence":0.9,"box":[10,10,80,40]}]
    progress=[]
    registry,info=process_video(path,pipeline,DEFAULTS,progress.append)
    assert len(registry.events)==1 and registry.events[0].image.startswith(b"\xff\xd8")
    assert progress[-1]==1 and registry.events[0].frame_number==0
