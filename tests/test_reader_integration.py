from dataclasses import replace
from pathlib import Path
import numpy as np
from config import DEFAULTS
from src.pipeline import Pipeline
from src.classical_detector import ClassicalDetector
from streamlit.testing.v1 import AppTest

class StubReader:
    def read(self,image):return "MH12AB1234",[.99]*10

def test_reader_crop_route_and_low_confidence():
    cfg=replace(DEFAULTS,reader_mode="whole_plate",localization_mode="crop",rectify=False)
    result=Pipeline(cfg,classifier=StubReader())(np.full((32,192,3),255,np.uint8))[0]
    assert result["recognized"] and result["box"]==[0,0,192,32]
    assert result["characters"]==[] and result["reader_mode"]=="whole_plate"

def test_blank_scene_has_no_classical_proposals():
    assert ClassicalDetector()(np.zeros((200,300),np.uint8))==[]

def test_ui_reader_modes():
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/"app.py"),default_timeout=30).run()
    assert not app.exception
    next(s for s in app.selectbox if s.label=="Plate reader").set_value("Whole-plate reader (experimental)").run()
    next(s for s in app.selectbox if s.label=="Find plates using").set_value("Already-cropped plate image").run()
    assert not app.exception
