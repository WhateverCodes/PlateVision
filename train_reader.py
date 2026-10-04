"""Two-minute resumable training session for the real-data plate reader."""
from pathlib import Path
from training.train_plate_reader import train
if __name__=="__main__":
    root=Path(__file__).resolve().parent
    print("PLATEVISION real-data reader: 2 CPU threads, batch 8, up to two minutes.")
    print("A current batch and checkpoint save can extend the time slightly.")
    train(root/"data/reader_v1",root/"models/plate_reader_pilot",seconds=120)
    print("Session saved. The app offers the reader as an experimental option.")
