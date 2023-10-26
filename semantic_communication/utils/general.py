import os
import random
from pathlib import Path

import torch
import numpy as np

RANDOM_STATE = 42


def set_seed():
    random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmarks = False
    torch.autograd.set_detect_anomaly(True)

# TODO: REMOVE
def get_device():
    return torch.device("cpu")   # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def print_loss(losses, group):
    mean_loss = np.mean(losses)
    se = np.std(losses, ddof=1) / np.sqrt(len(losses))
    print(f"{group} Mean Loss: {mean_loss:.3f} ± {se:.3f}")


def create_checkpoint(model, path, **kwargs):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    torch.save({"state_dict": model.state_dict(), **kwargs}, p)
