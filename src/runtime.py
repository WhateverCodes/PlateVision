"""Explicit resource limits and reproducibility helpers."""
import os
import random
import numpy as np

def configure(threads: int = 2, seed: int = 42) -> None:
    threads = max(1, min(threads, 4))
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    import torch
    import cv2
    torch.set_num_threads(threads)
    cv2.setNumThreads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
