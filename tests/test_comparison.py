from pathlib import Path
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from config import ROOT
from src.demo_media import example_path
from src.comparison import compare,reference_counts
from src.pretrained import PretrainedReader


def test_comparison_gives_both_routes_same_unmodified_pixels():
    class Pipe:
        def __init__(self):
            self.detector=self.classifier=self
            self.provenance='test';self.seen=None
        def __call__(self,image):
            self.seen=image.copy();image[:]=0
            return []
    first,second=Pipe(),Pipe();image=np.full((8,8,3),123,np.uint8)
    compare(image,{'own':first,'external':second})
    assert np.array_equal(first.seen,second.seen)
    assert (image==123).all()


def test_reference_scoring_counts_wrong_and_rejected_predictions():
    rows=[{'raw_text':'MH12AB1234','recognized':False},
          {'raw_text':'MH12AB1235','recognized':True}]
    assert reference_counts(rows,{'MH12AB1234'})=={'correct':0,'missed':1,'wrong':1}


def test_pretrained_preserves_repeated_characters_and_ignores_trailing_padding():
    class Model:
        def run(self,names,inputs):
            assert inputs['input'].dtype==np.uint8
            assert inputs['input'].shape==(1,2,3,3)
            # Source pixel is BGR; external model expects RGB.
            assert inputs['input'][0,0,0].tolist()==[30,20,10]
            return [np.array([[[.99,.01],[.98,.02],[.2,.8]]])]
    r=object.__new__(PretrainedReader);r.model=Model();r.input='input';r.output='plate'
    r.config={'img_width':3,'img_height':2,'alphabet':'1_','max_plate_slots':3,'pad_char':'_'}
    text,scores=r.read(np.full((2,3,3),[10,20,30],np.uint8))
    assert text=='11' and scores==[.99,.98]


@pytest.mark.skipif(example_path(ROOT) is None or not (ROOT/'models/pretrained/cct_s_v2_global.onnx').exists(),reason='Optional local example/model not bundled')
def test_app_comparison_runs_both_real_models():
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    next(r for r in app.radio if r.label=='Workspace').set_value('Compare models').run()
    next(c for c in app.checkbox if c.label=='Use comparison example').check().run()
    next(b for b in app.button if b.label=='Compare both models').click().run(timeout=60)
    assert not app.exception and not app.error
    assert len(app.dataframe)==2
    assert 'KL07BX7197' in app.dataframe[0].value['Raw reading'].tolist()
    # Changing input mode clears stale predictions before another run.
    next(r for r in app.radio if r.label=='What does the image contain?').set_value('One cropped plate').run()
    assert not app.dataframe
