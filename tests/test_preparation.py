import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from training.prepare_plate_dataset import prepare
from training.prepare_character_dataset import prepare as prepare_characters

def test_real_manifest_and_character_split_inheritance(tmp_path):
    records=[]
    for i in range(10):
        image=np.full((80,160,3),255,np.uint8)
        cv2.putText(image,"M",(20,60),cv2.FONT_HERSHEY_SIMPLEX,1.4,(0,0,0),2)
        image[0,0]=i
        path=tmp_path/f"{i}.png"
        cv2.imwrite(str(path),image)
        records.append({"image":path.name,"group":f"vehicle-{i}","boxes":[[10,10,150,70]],
                        "characters":[{"label":"M","box":[18,20,55,65]}]})
    manifest=tmp_path/"source.json"
    manifest.write_text(json.dumps({"source":"Generated unit test","license":"MIT","records":records}))
    prepare(manifest,tmp_path/"prepared",42)
    prepare_characters(tmp_path/"prepared",tmp_path/"chars","M")
    for split in ("train","val","test"):
        plates=json.loads((tmp_path/"prepared"/f"{split}.json").read_text())
        chars=json.loads((tmp_path/"chars"/f"{split}.json").read_text())
        assert {r["group"] for r in plates}=={r["group"] for r in chars}
        assert all(cv2.imread(str(tmp_path/"chars"/r["image"]),0).shape==(32,32) for r in chars)

def test_source_permissions_required(tmp_path):
    path=tmp_path/"source.json"
    path.write_text('{"records":[]}')
    with pytest.raises(ValueError,match="source and license"):
        prepare(path,tmp_path/"out",42)
