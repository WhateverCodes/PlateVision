from dataclasses import replace
import cv2
import numpy as np
from config import DEFAULTS
from src.pipeline import Pipeline
from src.timestamp_manager import Registry
from src.visualization import annotate

def plate():
    image=np.full((100,530,3),255,np.uint8)
    cv2.putText(image,"MH12AB1234",(15,75),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),3,cv2.LINE_AA)
    return image

def detector(gray):
    assert gray.ndim==2
    return [{"box":[0,0,530,100],"score":0.99}]

def test_pipeline_preprocessing_segmentation_and_reconstruction():
    # This is a unit test with an explicit stub classifier, not an accuracy claim.
    def classify(chars):
        assert len(chars)==10
        return "MH12AB1234",[0.99]*10
    pipe=Pipeline(replace(DEFAULTS,rectify=False),detector,classify)
    results=pipe(plate())
    assert results[0]["recognized"] and results[0]["text"]=="MH12AB1234"
    assert results[0]["vehicle_type"]=="Unknown"
    registry=Registry()
    registry.observe(results)
    drawn=annotate(plate(),results)
    assert results[0]["serial"]=="001" and drawn.shape==plate().shape

def test_low_confidence_not_invented():
    pipe=Pipeline(replace(DEFAULTS,rectify=False),detector,lambda chars:("MH12AB1234",[0.2]*10))
    result=pipe(plate())[0]
    assert result["text"]=="Not recognized" and not result["recognized"]
    assert Registry().observe([result]) is None
