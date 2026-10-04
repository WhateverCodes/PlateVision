import hashlib,json
from pathlib import Path
import torch
from PIL import Image
from config import ALPHABET
from src.plate_reader import PlateReader,decode
from src.runtime import configure
from training.train_plate_reader import train
from training.data_tools import grouped_split

def test_ctc_repetition_and_model_shape():
    configure(2)
    a=ALPHABET.index("A")+1
    values=torch.full((6,1,37),-20.)
    for i,c in enumerate([a,a,0,a,0,0]):values[i,0,c]=0
    assert decode(values)==["AA"]
    model=PlateReader();output=model(torch.zeros(2,1,32,192))
    assert output.shape==(48,2,37) and torch.isfinite(output).all()

def test_bounded_training_resumes(tmp_path):
    image=tmp_path/"image.png";Image.new("RGB",(100,32),"white").save(image)
    r={"image":str(image),"text":"AB12","file_sha256":hashlib.sha256(image.read_bytes()).hexdigest()}
    for s in ("train","val"):(tmp_path/(s+".json")).write_text(json.dumps([r,r]))
    out=tmp_path/"model"
    first=train(tmp_path,out,seconds=30,batch_size=1,max_steps=1)
    second=train(tmp_path,out,seconds=30,batch_size=1,max_steps=1)
    assert first["cursor"]==1 and second["cursor"]==2
    ck=torch.load(out/"last.pt",weights_only=True)
    assert ck["trained_steps"]==2 and not (out/"best.pt").exists()
    assert ck["state"]["partial_count"]==2


def test_reader_loads_both_architectures(tmp_path):
    from src.plate_reader import Reader
    import numpy as np
    for version in (1,2):
        model=PlateReader(version)
        checkpoint={"kind":"plate_reader","alphabet":ALPHABET,"model":model.state_dict(),"trained_steps":1,"architecture":version}
        path=tmp_path/(str(version)+".pt");torch.save(checkpoint,path)
        text,scores=Reader(path).read(np.full((32,192),255,np.uint8))
        assert isinstance(text,str) and len(text)==len(scores)
