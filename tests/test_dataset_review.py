import json
import pytest
from src.dataset_review import validate_review,save_review,load_reviews

def test_review_requires_confirmation_and_text(tmp_path):
    record={"id":"a"*20,"width":100,"height":50,"boxes":[[1,1,90,40]]}
    rows=[{"x1":1,"y1":1,"x2":90,"y2":40,"text":"MH12AB3456","unreadable":False}]
    with pytest.raises(ValueError): validate_review(record,rows,"vehicle-a","Plate close-up",False)
    result=validate_review(record,rows,"vehicle-a","Plate close-up",True)
    save_review(tmp_path,result)
    assert load_reviews(tmp_path)["a"*20]["texts"]==["MH12AB3456"]
    rows[0]["text"]=""
    with pytest.raises(ValueError): validate_review(record,rows,"vehicle-a","Plate close-up",True)
    rows[0]["unreadable"]=True
    assert validate_review(record,rows,"vehicle-a","Plate close-up",True)["texts"]==[""]
