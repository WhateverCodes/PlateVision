import json
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from streamlit.testing.v1 import AppTest


def test_glyph_correction_and_exclusion_save_and_advance(tmp_path):
    folder=tmp_path/'data/glyph_review';folder.mkdir(parents=True)
    image=tmp_path/'glyph.png';Image.new('RGB',(40,40),'white').save(image)
    rows=[{'id':str(i)*20+'_0','source_id':str(i)*20,'index':0,
           'image':str(image),'plate_image':str(image),'plate_text':'MH12AB1234',
           'suggested':'A','split':'train'} for i in [1,2]]
    (folder/'queue.json').write_text(json.dumps(rows))
    script=Path(__file__).resolve().parents[1]/'review_glyphs.py'
    with patch('config.ROOT',tmp_path):
        app=AppTest.from_file(str(script),default_timeout=30).run()
        assert not app.exception
        first=next(s for s in app.selectbox if s.label=='Character').value
        next(s for s in app.selectbox if s.label=='Actual character').set_value('R')
        app.button[0].click().run()
        assert not app.exception
        assert json.loads((folder/'reviews'/f'{first}.json').read_text())['label']=='R'
        assert next(s for s in app.selectbox if s.label=='Character').value!=first
        second=next(s for s in app.selectbox if s.label=='Character').value
        next(c for c in app.checkbox if c.key=='exclude_'+second).check()
        next(b for b in app.button if b.proto.form_id=='glyph_'+second).click().run()
        assert not app.exception
        saved=[json.loads(p.read_text()) for p in (folder/'reviews').glob('*.json')]
        assert len(saved)==2
        assert any(r['unusable'] and r['label']=='' for r in saved)
