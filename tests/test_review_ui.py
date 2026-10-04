import json
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from streamlit.testing.v1 import AppTest

def test_review_ui_saves_only_after_confirmation(tmp_path):
    folder=tmp_path/"data/kaggle_indian_review"
    (folder/"images").mkdir(parents=True)
    Image.new("RGB",(100,50),"white").save(folder/"images/example.png")
    record={"id":"a"*20,"image":"images/example.png","width":100,"height":50,
            "original_files":["images/example.png"],"boxes":[[1,1,90,40]],
            "texts":["MH12AB3456"],"status":"pending_review","reason":"",
            "suggested_kind":"plate_closeup","group":"test-vehicle"}
    (folder/"catalog.json").write_text(json.dumps([record]))
    (folder/"audit.json").write_text(json.dumps({"duplicate_copies_removed":0,"source":"Unit test","license":"MIT"}))
    with patch("config.ROOT",tmp_path):
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/"review_dataset.py"),default_timeout=30).run()
        assert not app.exception
        next(b for b in app.button if b.label=="Save review").click().run()
        assert app.error and not (folder/"reviews").exists()
        app.checkbox[0].check().run()
        next(b for b in app.button if b.label=="Save review").click().run()
        assert not app.exception
        saved=json.loads((folder/"reviews"/("a"*20+".json")).read_text())
        assert saved["status"]=="reviewed" and saved["texts"]==["MH12AB3456"]
