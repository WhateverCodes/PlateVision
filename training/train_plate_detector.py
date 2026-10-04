"""Run: python -m training.train_plate_detector --data data/synthetic/plates"""
from .engine import train
if __name__=="__main__":
    train("plate_detector")
