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


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def print_loss(losses, group):
    mean_loss = np.mean(losses)
    se = np.std(losses, ddof=1) / np.sqrt(len(losses))
    print(f"{group} Mean Loss: {mean_loss:.3f} ± {se:.3f}")


def create_checkpoint(path, **kwargs):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    torch.save({**kwargs}, p)


def add_semantic_decoder_args(parser):
    parser.add_argument("--n-blocks", default=1, type=int)
    parser.add_argument("--n-heads", default=4, type=int)
    parser.add_argument("--n-embeddings", default=384, type=int)


def add_train_args(parser):
    parser.add_argument("--n-epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--checkpoint-path", default="checkpoints", type=str)


def add_data_args(parser):
    parser.add_argument("--max-length", default=30, type=int)
    parser.add_argument("--data-fp", default="", type=str)


def add_channel_model_args(parser):
    parser.add_argument("--channel-block-input-dim", default=384, type=int)
    parser.add_argument("--channel-block-latent-dim", default=128, type=int)

    parser.add_argument("--sig-pow", default=1.0, type=float)
    parser.add_argument("--SNR-min", default=3, type=int)
    parser.add_argument("--SNR-max", default=21, type=int)
    parser.add_argument("--SNR-step", default=3, type=int)
    parser.add_argument("--SNR-window", default=5, type=int)
    parser.add_argument("--SNR-diff", default=3, type=int)
    parser.add_argument("--channel-type", default="AWGN", type=str)
