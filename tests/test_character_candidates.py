import numpy as np
from src.character_classifier import Classifier

class FakeClassifier(Classifier):
    def __init__(self,predictions):self.predictions=predictions
    def __call__(self,chars):return self.predictions[chars[0]]

def test_low_confidence_alternative_cannot_displace_baseline(monkeypatch):
    mask=np.zeros((2,2),dtype=np.uint8)
    monkeypatch.setattr("src.character_segmenter.segmentation_candidates",lambda image:[(10,[0],mask),(9,[1],mask)])
    c=FakeClassifier([("UK106679",[.9]*8),("JX10BF79",[.3]*8)])
    assert c.read_segmented(mask)[2]=="UK106679"

def test_high_confidence_plausible_alternative(monkeypatch):
    mask=np.zeros((2,2),dtype=np.uint8)
    monkeypatch.setattr("src.character_segmenter.segmentation_candidates",lambda image:[(12,[0],mask),(10,[1],mask)])
    c=FakeClassifier([("INDMH12AB1234",[.99]*13),("MH12AB1234",[.99]*10)])
    assert c.read_segmented(mask)[2]=="MH12AB1234"
