import numpy as np
from streamlit.testing.v1 import AppTest
from config import ROOT, DEFAULTS
from src.comparison import compare, build_pipelines


def test_comparison_preserves_rejected_candidates_and_sorts_reading_scores():
    class Pipe:
        provenance='test'
        def __init__(self):
            self.detector=self.classifier=self
        def __call__(self,image):
            return [dict(confidence=.2,recognized=False),dict(confidence=.9,recognized=True),dict(confidence=.6,recognized=False)]
    rows=compare(np.zeros((10,10,3),np.uint8),{'own':Pipe()})['own']['detections']
    assert [r['confidence'] for r in rows]==[.9,.6,.2]
    assert len(rows)==3


def test_normal_comparison_shares_pretrained_localization(monkeypatch):
    import src.comparison as module
    detector=object()
    monkeypatch.setattr(module,'PretrainedDetector',lambda *args:detector)
    monkeypatch.setattr(module,'WholeLineReader',lambda *args:object())
    class Pipe:
        def __init__(self,settings,detector=None,classifier=None):
            self.settings,self.detector=settings,detector
    monkeypatch.setattr(module,'Pipeline',Pipe)
    pipes=build_pipelines(DEFAULTS)
    assert all(p.detector is detector for p in pipes.values())
    assert pipes['My model'].settings.reader_mode=='characters'
    assert pipes['Pretrained model'].settings.reader_mode=='whole_line'


def test_main_workspace_has_no_model_or_processing_choices():
    app=AppTest.from_file(str(ROOT/'app.py')).run()
    assert not app.exception
    assert not app.selectbox and not app.slider
    assert not any('Training guide' in e.label for e in app.expander)
