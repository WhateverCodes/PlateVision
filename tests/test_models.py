import torch
import pytest
from src.architectures import PlateDetector,CharacterClassifier
from src.plate_detector import load_checkpoint
from src.runtime import configure
from training.losses import detection_loss
from training.evaluate_plate_detector import metrics

configure()

def test_detector_multiple_scales_and_gradients():
    model=PlateDetector()
    result=model(torch.rand(2,1,64,64))
    assert [x[0].shape[-2:] for x in result]==[(8,8),(4,4),(2,2)]
    loss=detection_loss(result,[torch.tensor([[8.,8.,40.,30.]]),torch.empty((0,4))])
    assert torch.isfinite(loss)
    loss.backward()
    assert model.box.weight.grad.abs().sum()>0

def test_character_model_gradient():
    model=CharacterClassifier()
    logits=model(torch.rand(2,1,32,32))
    assert logits.shape==(2,36)
    loss=torch.nn.functional.cross_entropy(logits,torch.tensor([0,35]))
    loss.backward()
    assert model.head[-1].weight.grad.abs().sum()>0

def test_missing_model_error(tmp_path):
    with pytest.raises(ValueError,match="Missing"):
        load_checkpoint(tmp_path/"none.pt","plate_detector")

def test_untrained_model_rejected(tmp_path):
    path=tmp_path/"model.pt"
    torch.save({"format_version":1,"kind":"plate_detector","trained_steps":0},path)
    with pytest.raises(ValueError,match="not been trained"):
        load_checkpoint(path,"plate_detector")

def test_metrics_count_duplicates_and_misses():
    predictions=[[{"box":[0,0,10,10],"score":0.9},{"box":[0,0,10,10],"score":0.8}],[]]
    truth=[[[0,0,10,10]],[[20,20,30,30]]]
    result=metrics(predictions,truth)
    assert result["precision"]==0.5 and result["recall"]==0.5
    assert 0.49<result["ap"]<0.52
