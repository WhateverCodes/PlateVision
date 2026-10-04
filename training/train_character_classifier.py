"""Run: python -m training.train_character_classifier --data data/synthetic/characters"""
from .engine import train
if __name__=="__main__":
    train("character_classifier")
