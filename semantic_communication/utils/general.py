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


def load_model(model, state_dict_path):
    if state_dict_path is not None:
        checkpoint = torch.load(state_dict_path, map_location=get_device())
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"{state_dict_path} is loaded.")
    else:
        print("state_dict_path is None!")


def load_optimizer(optimizer, state_dict_path):
    if state_dict_path is not None:
        checkpoint = torch.load(state_dict_path, map_location=get_device())
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def add_semantic_decoder_args(parser):
    parser.add_argument("--n-blocks", default=1, type=int)
    parser.add_argument("--n-heads", default=4, type=int)
    parser.add_argument("--n-embeddings", default=384, type=int)
    parser.add_argument("--mode", default="predict", type=str)


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
    parser.add_argument("--channel-block-latent-dim", default=256, type=int)

    parser.add_argument("--channel-type", default="AWGN", type=str)
    parser.add_argument("--alpha", default=4.0, type=float)
    parser.add_argument("--sig-pow", default=1.0, type=float)
    parser.add_argument("--noise-pow", default=4e-15, type=float)
    parser.add_argument("--d-min", default=2e3, type=float)
    parser.add_argument("--d-max", default=7e3, type=float)
    parser.add_argument("--gamma-min", default=0.2, type=float)
    parser.add_argument("--gamma-max", default=0.8, type=float)


def shift_inputs(xb, encoder_output, attention_mask, mode):
    if mode == "predict":
        idx = xb[:, :-1]
        encoder_output = encoder_output[:, :-1, :]
        attention_mask = attention_mask[:, :-1]
        targets = xb[:, 1:]
    elif mode == "forward":
        idx = xb[:, :-1]
        encoder_output = encoder_output[:, 1:, :]
        attention_mask = attention_mask[:, 1:]
        targets = xb[:, 1:]
    else:
        raise ValueError("Mode needs to be 'predict' or 'forward'.")

    return idx, encoder_output, attention_mask, targets
