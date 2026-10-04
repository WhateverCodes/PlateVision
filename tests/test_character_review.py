import json
from pathlib import Path
from unittest.mock import patch
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest
from src.character_review import save_characters,fingerprint

def record():
    return {"id":"a"*20,"sha256":"sample","image":"images/example.png","group":"test",
            "original_files":["example.png"],"boxes":[[10,5,90,45]],"texts":["AB"],"status":"reviewed"}

def test_coordinates_and_incomplete_review(tmp_path):
    r=record()
    with pytest.raises(ValueError): save_characters(tmp_path,r,0,[None,None])
    entries=[{"label":c,"box":[i*30,0,i*30+20,30],"polarity":"dark","checked":True} for i,c in enumerate("AB")]
    target=save_characters(tmp_path,r,0,entries)
    result=json.loads(target.read_text())
    assert result["characters"][0]["box"]==[10,5,30,35]
    before=fingerprint(r,0)
    r["texts"]=["AC"]
    assert before!=fingerprint(r,0)
    with pytest.raises(ValueError): save_characters(tmp_path,r,0,entries)

def test_character_ui_confirm_save_and_invalidate(tmp_path):
    folder=tmp_path/"data/kaggle_indian_review"
    (folder/"images").mkdir(parents=True)
    (folder/"reviews").mkdir()
    Image.new("RGB",(100,50),"white").save(folder/"images/example.png")
    r=record()
    (folder/"reviews"/(r["id"]+".json")).write_text(json.dumps(r))
    with patch("config.ROOT",tmp_path):
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/"review_characters.py"),default_timeout=30).run()
        assert not app.exception
        assert app.button[1].disabled
        app.slider[0].set_value((0,20)).run()
        app.button[0].click().run()
        next(s for s in app.selectbox if s.label=="Character to check").set_value(1).run()
        app.slider[0].set_value((30,50)).run()
        app.button[0].click().run()
        assert not app.button[1].disabled
        app.button[1].click().run()
        target=folder/"character_reviews"/(r["id"]+"_0.json")
        assert json.loads(target.read_text())["characters"][1]["box"]==[40,5,60,45]
        app.slider[0].set_value((31,50)).run()
        assert app.button[1].disabled
        assert not app.exception
