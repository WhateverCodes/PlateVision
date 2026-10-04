import io,json
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import patch
from PIL import Image
import pytest
from streamlit.testing.v1 import AppTest
from training.import_ocr_dataset import import_dataset

def make_zip(path,unsafe=False):
    buf=io.BytesIO();Image.new("RGB",(100,20),"white").save(buf,format="JPEG")
    with ZipFile(path,"w") as z:
        for split,text in [("train","MH4705851"),("valid","MH4705851"),("test","blur")]:
            z.writestr(split+"_labels.txt","plate.jpg\t"+text+"\n")
            z.writestr("images/"+split+"/plate.jpg",buf.getvalue())
        if unsafe:z.writestr("../bad.txt","bad")

def test_import_preserves_split_identity_and_placeholders(tmp_path):
    archive=tmp_path/"input.zip";make_zip(archive);out=tmp_path/"out"
    audit=import_dataset(archive,out)
    rows=json.loads((out/"catalog.json").read_text())
    assert len({r["id"] for r in rows})==3
    assert audit["reviewable"]==2 and audit["excluded_placeholders"]==1
    assert audit["source_texts_across_splits"]==1
    assert not (out/"train.json").exists()
    with pytest.raises(ValueError):import_dataset(archive,out)
    make_zip(tmp_path/"unsafe.zip",True)
    with pytest.raises(ValueError):import_dataset(tmp_path/"unsafe.zip",tmp_path/"bad")

def test_review_correction_persists_and_next_advances(tmp_path):
    archive=tmp_path/"input.zip";make_zip(archive)
    folder=tmp_path/"data/ocr_text_review";import_dataset(archive,folder)
    with patch("config.ROOT",tmp_path):
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/"review_ocr_labels.py"),default_timeout=30).run()
        assert not app.exception
        identity=next(s for s in app.selectbox if s.label=="Image").value
        app.text_input[0].set_value("MH47Q5851")
        app.button[0].click().run()
        saved=json.loads((folder/"reviews"/(identity+".json")).read_text())
        assert saved["text"]=="MH47Q5851" and saved["source_text"]=="MH4705851"
        assert next(s for s in app.selectbox if s.label=="Image").value!=identity
        assert not saved["group_verified"]
        app.checkbox[0].check();app.button[0].click().run()
        assert len(list((folder/"reviews").glob("*.json")))==2
        assert not app.exception
