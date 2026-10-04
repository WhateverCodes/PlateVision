import io,json,zipfile
import numpy as np
import pytest
from PIL import Image
from training.import_kaggle_yolo import yolo_boxes,safe_members,import_dataset

def test_normalized_boxes_convert_to_pixels_and_multiple_plates():
    boxes=yolo_boxes("0 .5 .5 .4 .2\n0 .2 .2 .2 .2",200,100)
    assert boxes==[[20.,10.,60.,30.],[60.,40.,140.,60.]]

@pytest.mark.parametrize("text",["0 nan .5 .2 .2","1 .5 .5 .2 .2","0 .99 .5 .5 .2","0 .5 .5 -.2 .2"])
def test_invalid_boxes_rejected(text):
    with pytest.raises(ValueError): yolo_boxes(text,200,100)

def test_zip_traversal_rejected():
    data=io.BytesIO()
    with zipfile.ZipFile(data,"w") as z: z.writestr("../outside.png",b"bad")
    with zipfile.ZipFile(data) as z:
        with pytest.raises(ValueError): safe_members(z)

def test_dedup_and_missing_labels_not_negative(tmp_path):
    images=[]
    for value in [70,160]:
        stream=io.BytesIO()
        Image.new("RGB",(100,40),(value,20,30)).save(stream,format="PNG")
        images.append(stream.getvalue())
    archive=tmp_path/"source.zip"
    with zipfile.ZipFile(archive,"w") as z:
        z.writestr("images/a.png",images[0]);z.writestr("images/b.png",images[0])
        z.writestr("labels/a.txt","0 .5 .5 .8 .8");z.writestr("labels/b.txt","0 .5 .5 .8 .8")
        z.writestr("images/c.png",images[1])
    out=tmp_path/"review"
    report=import_dataset(archive,out)
    assert report["duplicate_copies_removed"]==1
    assert report["images_with_usable_boxes"]==1
    assert report["quarantined_unique_images"]==1
    assert not (out/"train.json").exists()
    with pytest.raises(ValueError,match="already"):
        import_dataset(archive,out)
