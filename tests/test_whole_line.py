from dataclasses import replace
import numpy as np
from config import DEFAULTS
from src.whole_line_reader import clean_line,order_lines
from src.pipeline import Pipeline
from src.whole_line_reader import WholeLineReader
from src.paddle_candidate import PaddleCandidate


def test_adaptation_provenance_uses_checkpoint_metadata(tmp_path,monkeypatch):
    import json
    import src.whole_line_reader as module
    class Session:
        def get_inputs(self):
            return [type('Input',(),{'name':'image'})()]
    monkeypatch.setattr(module,'session',lambda *args:Session())
    monkeypatch.setattr(module,'PaddleCandidate',lambda *args:object())
    (tmp_path/'adaptation.json').write_text(json.dumps({'training_lines':981,'label_description':'mixed-label training lines'}))
    reader=WholeLineReader(tmp_path)
    assert '981 mixed-label training lines' in reader.provenance
    assert '76 reviewed' not in reader.provenance


def test_slanted_single_row_keeps_left_to_right_order():
    items=[dict(cx=160,cy=20,height=20,text='8008'),
           dict(cx=10,cy=65,height=20,text='HR26'),
           dict(cx=90,cy=41,height=20,text='BA')]
    assert ''.join(r['text'] for r in order_lines(items))=='HR26BA8008'


def test_retry_does_not_replace_disagreeing_text():
    reader=WholeLineReader.__new__(WholeLineReader)
    reader._read_once=lambda image: ('MH01AB3456',[.88]*10) if image.shape[0]==20 else ('MH01AB3457',[.99]*10)
    assert reader.read(np.zeros((20,80,3),np.uint8))==('MH01AB3456',[.88]*10)
    reader._read_once=lambda image: ('MH01AB3456',[.88 if image.shape[0]==20 else .99]*10)
    assert reader.read(np.zeros((20,80,3),np.uint8))[1]==[.99]*10


def test_tiled_detector_maps_coordinates_and_removes_duplicates():
    from src.pretrained import PretrainedDetector
    detector=PretrainedDetector.__new__(PretrainedDetector)
    detector._detect=lambda image:[{'box':[10,10,50,30],'score':.9}]
    rows=detector(np.zeros((800,1000,3),np.uint8))
    assert len(rows)==4
    assert [410,330,450,350] in [r['box'] for r in rows]


def fake_reader(alternatives):
    reader=WholeLineReader.__new__(WholeLineReader)
    reader.line_crops=lambda image: []
    class Tokens:
        def alternatives(self,image):return alternatives
    reader.reader=Tokens()
    return reader


def test_format_repair_uses_model_alternative_and_keeps_its_scores():
    reader=fake_reader([('MHO1AB3456',[.99]*10),
                        ('MH01AB3456',[.99,.99,.12]+[.99]*7)])
    text,scores=reader.read(np.zeros((20,80,3),np.uint8))
    assert text=='MH01AB3456' and scores[2]==.12
    reader=fake_reader([('MH01AB3456',[.99]*10),('MH01AB3457',[.9]*10)])
    assert reader.read(np.zeros((20,80,3),np.uint8))[0]=='MH01AB3456'


def test_removed_separator_still_counts_against_candidate_likelihood():
    reader=fake_reader([('MH02CD3654',[.98]*10),
                        ('MH02CD 654',[.999]*6+[.02]+[.999]*3)])
    assert reader.read(np.zeros((20,80,3),np.uint8))[0]=='MH02CD3654'


def test_token_alternatives_preserve_blank_separated_repeats():
    reader=PaddleCandidate.__new__(PaddleCandidate)
    reader.alphabet=['','A','B']
    reader.predict_tokens=lambda image: np.array([[.01,.95,.04],[.01,.95,.04],
                                                  [.98,.01,.01],[.01,.95,.04]])
    options=reader.alternatives(None)
    assert options[0][0]=='AA' and len(options[0][1])==2


def test_two_line_registration_orders_top_then_bottom():
    items=[dict(cx=80,cy=65,height=25,text='3456'),dict(cx=15,cy=25,height=25,text='MH12'),
           dict(cx=10,cy=65,height=25,text='AB')]
    assert ''.join(r['text'] for r in order_lines(items))=='MH12AB3456'


def test_separator_cleanup_keeps_scores_aligned_and_does_not_guess_letters():
    text,scores=clean_line('M.H 01-AO',[.9]*10)
    assert text=='MH01AO' and len(scores)==6


def test_whole_line_pipeline_never_cuts_characters(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Character segmentation must not run')
    monkeypatch.setattr('src.pipeline.segment',forbidden)
    class Detector:
        def __call__(self,image):
            assert image.ndim==3
            return [{'box':[0,0,50,25],'score':.9}]
    class Reader:
        def read(self,image):return 'MH12AB3456',[.99]*10
    cfg=replace(DEFAULTS,reader_mode='whole_line',localization_mode='pretrained',rectify=False,character_threshold=.95)
    result=Pipeline(cfg,Detector(),Reader())(np.full((25,50,3),180,np.uint8))[0]
    assert result['recognized'] and result['text']=='MH12AB3456'
    assert result['characters']==[] and result['confidence_method']=='mean OCR token score'
