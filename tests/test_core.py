import cv2
import numpy as np
import pytest
from src.preprocessing import grayscale,letterbox,normalize_character,read_image
from src.character_segmenter import Character,order_characters,segment
from src.validator import normalize,plausible,same_plate
from src.timestamp_manager import Registry,timestamp
from src.association import associate

def detection(text):
    return {"text":text,"recognized":True,"confidence":0.95,"vehicle_type":"Car"}

def test_grayscale_and_no_mutation():
    image = np.array([[[0,0,255],[0,255,0]]],dtype=np.uint8)
    original = image.copy()
    result = grayscale(image)
    assert result.tolist() == [[76,150]]
    assert np.array_equal(image,original)

def test_letterbox():
    out,scale,offset=letterbox(np.ones((100,200),np.uint8),320)
    assert out.shape==(320,320) and scale==1.6 and offset==(0,80)

def test_invalid_media():
    with pytest.raises(ValueError):
        read_image(b"broken")
    with pytest.raises(ValueError):
        grayscale(np.array([]))

def test_char_normalization():
    mask=np.zeros((60,30),np.uint8)
    mask[10:50,12:18]=255
    out=normalize_character(mask)
    assert out.shape==(32,32) and out[:,14:18].sum()>0
    assert out[:4].sum()==0

def test_order_two_rows():
    dummy=np.zeros((32,32),np.uint8)
    chars=[Character((40,50,50,80),dummy),Character((30,0,40,30),dummy),
           Character((10,50,20,80),dummy),Character((10,0,20,30),dummy)]
    assert [c.box[:2] for c in order_characters(chars)]==[(10,0),(30,0),(10,50),(40,50)]

def test_segment_real_raster_glyphs():
    plate=np.full((100,530),255,np.uint8)
    cv2.putText(plate,"MH12AB1234",(15,75),cv2.FONT_HERSHEY_SIMPLEX,2,0,3,cv2.LINE_AA)
    chars,_=segment(plate)
    assert len(chars)==10
    assert all(c.image.shape==(32,32) for c in chars)

def test_normalize_without_inventing_characters():
    assert normalize(" mh-12 ab 1234 ")=="MH12AB1234"
    assert normalize("IO01")=="IO01"
    assert plausible("MH12AB1234")
    assert plausible("22BH1234AA")
    assert not plausible("HELLO")
    assert not plausible("MH12@B1234")

def test_duplicate_safety():
    assert same_plate("MH12AB1234","MH12AB1234")
    assert not same_plate("MH12AB1234","MH12AB1235",0.8)
    assert not same_plate("MH12AB1234","MH12AC1234")
    assert same_plate("MH12AB1234","MH12AC1234",0.9)

def test_one_event_per_frame():
    registry=Registry()
    first=[detection("MH12AB1234"),detection("KA01XY5678")]
    event=registry.observe(first,5,100,20)
    assert len(registry.events)==1 and event.serials==["001","002"]
    assert first[0]["serial"]=="001"
    assert registry.observe([detection("MH12AB1234"),detection("KA01XY5678")],6,120,20) is None
    event=registry.observe([detection("DL05CD4321")],12.5,250,20)
    assert len(registry.events)==2 and event.new_serials==["003"]
    assert len(registry.rows(True,0))==2
    assert len(registry.rows(True,1))==1
    assert "Timestamp" not in registry.rows()[0]

def test_event_keeps_old_plate_with_new_and_reset():
    r=Registry()
    r.observe([detection("MH12AB1234")],1)
    ev=r.observe([detection("MH12AB1234"),detection("DL05CD4321")],2)
    assert ev.serials==["001","002"] and ev.new_serials==["002"]
    r.reset()
    assert not r.entries and not r.events
    d=detection("KA01XY5678")
    r.observe([d])
    assert d["serial"]=="001"

def test_unrecognized_not_registered():
    assert Registry().observe([{"text":"Not recognized","recognized":False}]) is None

def test_timestamp():
    assert timestamp(83.9)=="01:23"

def test_association_does_not_guess_distant_vehicle():
    v={"box":[0,0,100,100],"vehicle_type":"Car"}
    assert associate([30,70,60,90],[v])==v
    assert associate([500,500,530,520],[v]) is None
