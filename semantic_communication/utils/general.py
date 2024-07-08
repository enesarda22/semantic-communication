import os
import random
import warnings
from pathlib import Path

import torch
import numpy as np

import matplotlib.pyplot as plt
import scienceplots

RANDOM_STATE = 42


def valid_mode(mode):
    return mode in ["sentence", "forward", "predict", "next_sentence"]


def set_seed(offset=0):
    random.seed(RANDOM_STATE + offset)
    torch.manual_seed(RANDOM_STATE + offset)
    torch.cuda.manual_seed(RANDOM_STATE + offset)
    np.random.seed(RANDOM_STATE + offset)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmarks = False
    torch.autograd.set_detect_anomaly(True)


def get_device():
    if "LOCAL_RANK" in os.environ:  # ddp
        rank = int(os.environ["LOCAL_RANK"])
        return torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def round_to_nearest_even(num):
    rounded = round(num)
    return (
        rounded if rounded % 2 == 0 else rounded + 1 if num >= rounded else rounded - 1
    )


def round_to_nearest_even(num):
    rounded = round(num)
    return (
        rounded if rounded % 2 == 0 else rounded + 1 if num >= rounded else rounded - 1
    )


def round_to_nearest_even(num):
    rounded = round(num)
    return (
        rounded if rounded % 2 == 0 else rounded + 1 if num >= rounded else rounded - 1
    )


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
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict=checkpoint["model_state_dict"],
            strict=False,
        )
        if len(missing_keys) > 0:
            raise RuntimeError(f"{missing_keys} are missing!")
        elif len(unexpected_keys) > 0:
            warnings.warn(f"{unexpected_keys} are unexpected!")
        else:
            print(f"{state_dict_path} is loaded.")
    else:
        print("state_dict_path is None!")


def load_optimizer(optimizer, state_dict_path):
    if state_dict_path is not None:
        checkpoint = torch.load(state_dict_path, map_location=get_device())
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def load_scheduler(scheduler, state_dict_path):
    if state_dict_path is not None:
        checkpoint = torch.load(state_dict_path, map_location=get_device())
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])


def get_start_epoch(state_dict_path):
    if state_dict_path is not None:
        checkpoint = torch.load(state_dict_path, map_location=get_device())
        return checkpoint["epoch"] + 1
    else:
        return 1


def add_semantic_decoder_args(parser):
    parser.add_argument("--n-blocks", default=1, type=int)
    parser.add_argument("--n-heads", default=4, type=int)
    parser.add_argument("--n-embeddings", default=384, type=int)
    parser.add_argument("--mode", default="predict", type=str)
    parser.add_argument("--rate", default=1, type=int)


def add_train_args(parser):
    parser.add_argument("--n-epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--eval-iter", default=None, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--checkpoint-path", default="checkpoints", type=str)
    parser.add_argument("--load-optimizer", default=False, type=bool)
    parser.add_argument("--load-scheduler", default=False, type=bool)


def add_data_args(parser):
    parser.add_argument("--max-length", default=30, type=int)
    parser.add_argument("--data-fp", default="", type=str)


def add_channel_model_args(parser):
    parser.add_argument("--channel-block-input-dim", default=384, type=int)
    parser.add_argument("--channel-block-latent-dim", default=64, type=int)

    parser.add_argument("--channel-type", default="", type=str)
    parser.add_argument("--alpha", default=4.0, type=float)
    parser.add_argument("--sig-pow", default=1.0, type=float)
    parser.add_argument("--noise-pow", default=4e-15, type=float)
    parser.add_argument("--d-min", default=2e3, type=float)
    parser.add_argument("--d-max", default=7e3, type=float)
    parser.add_argument("--gamma-min", default=0.2, type=float)
    parser.add_argument("--gamma-max", default=0.8, type=float)


def pad_cls(ids):
    return torch.cat(
        [
            torch.ones(ids.shape[0], 1, dtype=torch.int64, device=get_device()),
            ids,
        ],
        dim=1,
    )


def shift_inputs(xb, attention_mask, mode):
    assert valid_mode(mode)

    if mode == "predict":
        idx = xb[:, :-1]
        targets = xb[:, 1:]
        enc_padding_mask = attention_mask[:, 1:] == 0  # mask the end token
        is_causal = True
    elif mode == "forward":
        idx = xb[:, :-1]
        targets = xb[:, 1:]
        enc_padding_mask = attention_mask[:, 1:] == 0  # CLS is not received
        is_causal = True
    elif mode == "sentence" or mode == "next_sentence":
        idx = xb[:, :-1]
        targets = xb[:, 1:]
        enc_padding_mask = None
        is_causal = False
    else:
        raise ValueError("Mode needs to be 'predict', 'forward' or 'sentence'.")

    return idx, targets, enc_padding_mask, is_causal


def split_string_by_lengths(input_string, lengths):
    words = input_string.split()
    sentences = []
    current_index = 0

    for length in lengths:
        sentence = words[current_index : current_index + length]
        sentences.append(" ".join(sentence))
        current_index += length

    return sentences


def plotter(
    x_axis,
    xlabel,
    ylabel,
    title,
    separation_conventional=None,
    SPF=None,
    SLF=None,
    sentence_decode=None,
    sentence_predict=None,
    AE_baseline=None,
    save=True,
    show=False,
):
    plt.style.use(["science", "ieee", "no-latex"])
    plt.figure(figsize=(3.2, 2.6))

    lw = 1.3
    if not np.all(separation_conventional == 0):
        plt.plot(x_axis, separation_conventional, label="Conv. Baseline", linewidth=lw)
    if not np.all(SPF == 0):
        plt.plot(x_axis, SPF, label="SPF", linewidth=lw)
    if not np.all(SLF == 0):
        plt.plot(x_axis, SLF, label="SLF", linewidth=lw)
    if not np.all(AE_baseline == 0):
        plt.plot(x_axis, AE_baseline, label="AE Baseline", linewidth=lw)
    if not np.all(sentence_decode == 0):
        plt.plot(
            x_axis,
            sentence_decode,
            label="Sen. DF",
            color="c",
            linestyle=(0, (5, 1, 1, 1, 1, 1)),
            linewidth=lw,
        )
    if not np.all(sentence_predict == 0):
        plt.plot(
            x_axis,
            sentence_predict,
            label="Sen. PF",
            color="m",
            linestyle=(0, (3, 1, 1, 5, 1, 1)),
            linewidth=lw,
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([np.min(x_axis), np.max(x_axis)])
    plt.grid(lw=0.2)
    plt.legend()
    plt.title(title)
    plt.tight_layout()

    if save:
        plots_dir = "Plots"
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        file_path = os.path.join("Plots", f"{title}.png")
        plt.savefig(file_path, dpi=900)

    if show:
        plt.show()


def plot(
    d_sd_list,
    y_label,
    gamma_list,
    separation_conventional=None,
    SPF=None,
    SLF=None,
    sentence_decode=None,
    sentence_predict=None,
    AE_baseline=None,
    save=True,
    show=False,
):

    if separation_conventional is None:
        separation_conventional = np.zeros((len(d_sd_list), len(gamma_list)))
    if SPF is None:
        SPF = np.zeros((len(d_sd_list), len(gamma_list)))
    if SLF is None:
        SLF = np.zeros((len(d_sd_list), len(gamma_list)))
    if sentence_decode is None:
        sentence_decode = np.zeros((len(d_sd_list), len(gamma_list)))
    if sentence_predict is None:
        sentence_predict = np.zeros((len(d_sd_list), len(gamma_list)))
    if AE_baseline is None:
        AE_baseline = np.zeros((len(d_sd_list), len(gamma_list)))

    # for all distances
    for index, d_sd in enumerate(d_sd_list):
        plotter(
            x_axis=np.array(gamma_list) * d_sd,
            separation_conventional=separation_conventional[index, :],
            SPF=SPF[index, :],
            SLF=SLF[index, :],
            sentence_decode=sentence_decode[index, :],
            sentence_predict=sentence_predict[index, :],
            AE_baseline=AE_baseline[index, :],
            save=save,
            xlabel="$d_{sr}$ (m)",
            ylabel=y_label,
            title=f"$d_{{sr}}$ v. {y_label} for $d_{{sd}}={d_sd}$m",
            show=show,
        )

    mid_index = gamma_list.index(0.5)

    plotter(
        x_axis=d_sd_list,
        separation_conventional=separation_conventional[:, mid_index],
        SPF=SPF[:, mid_index],
        SLF=SLF[:, mid_index],
        sentence_decode=sentence_decode[:, mid_index],
        sentence_predict=sentence_predict[:, mid_index],
        AE_baseline=AE_baseline[:, mid_index],
        save=save,
        xlabel="$d_{sr}$ (m)",
        ylabel=y_label,
        title=f" $d_{{sd}}$ v. {y_label} for $d_{{sr}}=0.5d_{{sd}}$",
        show=show,
    )
