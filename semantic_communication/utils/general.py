import os
import random
import warnings
from pathlib import Path
import matplotlib as mpl

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
    parser.add_argument("--state-memory-len", default=-1, type=int)


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
    title=None,  # UPDATED: optional (so we can use a single fig-level title)
    separation_conventional=None,
    SPF=None,
    SLF=None,
    sentence_decode=None,
    sentence_predict=None,
    AE_baseline=None,
    LLM_baseline=None,
    ax=None,          # UPDATED: draw on provided axes (for subplots)
    legend=False,     # UPDATED: per-axis legend off by default (we use shared legend)
):
    # --- Style ---
    plt.style.use(["science", "ieee", "no-latex"])
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "STIXGeneral"],
        "mathtext.fontset": "stix",
    })

    if ax is None:
        _, ax = plt.subplots(figsize=(3.0, 2.6))

    lw = 1.5
    handles = []

    def _plot_if_valid(y, *args, **kwargs):
        if y is None:
            return
        y = np.asarray(y)
        if y.size == 0 or np.all(y == 0):
            return
        (h,) = ax.plot(x_axis, y, *args, **kwargs)
        handles.append(h)

    # --- Curves (colored, not BW-only) ---
    _plot_if_valid(separation_conventional, label="Conv. Baseline", linewidth=lw)

    _plot_if_valid(
        LLM_baseline,
        label="LLM Conv. Baseline",
        color="m",
        linestyle=(0, (3, 1, 1, 5, 1, 1)),
        linewidth=lw,
    )

    _plot_if_valid(SPF, label="SPF", linewidth=lw)
    _plot_if_valid(SLF, label="SLF", linewidth=lw)
    _plot_if_valid(AE_baseline, label="AE-JSCC", linewidth=lw)

    _plot_if_valid(
        sentence_decode,
        label="SSF",
        color="c",
        linestyle=(0, (5, 1, 1, 1, 1, 1)),
        linewidth=lw,
    )

    _plot_if_valid(
        sentence_predict,
        label="Sen. PF",
        color="m",
        linestyle=(0, (3, 1, 1, 5, 1, 1)),
        linewidth=lw,
    )

    # --- Axes styling ---
    ax.set_xlabel(xlabel, fontsize=11.2)
    ax.set_ylabel(ylabel, fontsize=11.2)
    ax.set_xlim([float(np.min(x_axis)), float(np.max(x_axis))])
    ax.grid(True, lw=0.2)

    if title:  # UPDATED: only set per-panel title if provided
        ax.set_title(title)

    if legend:
        h_, l_ = ax.get_legend_handles_labels()
        ax.legend(h_, l_, ncols=len(l_), frameon=True, fancybox=True, fontsize=11.2)

    labels = [h.get_label() for h in handles]
    return handles, labels


def plot_three_metrics_figure(
    x_axis,
    x_label,
    fig_title,   # UPDATED: one title for the whole 1x3 figure
    # y-axis labels for the 3 panels
    y1_label="BLEU Score",
    y2_label="SBERT Score",
    y3_label="GPT Score",
    # metric 1 arrays (1D)
    m1_sep=None, m1_spf=None, m1_slf=None, m1_ssf=None, m1_ae=None, m1_llm=None,
    # metric 2 arrays (1D)
    m2_sep=None, m2_spf=None, m2_slf=None, m2_ssf=None, m2_ae=None, m2_llm=None,
    # metric 3 arrays (1D)
    m3_sep=None, m3_spf=None, m3_slf=None, m3_ssf=None, m3_ae=None, m3_llm=None,
    out_path="Plots/three_metrics.pdf",
    show=False,
):
    plt.style.use(["science", "ieee", "no-latex"])
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "STIXGeneral"],
        "mathtext.fontset": "stix",
    })

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.6))

    # UPDATED: one title for entire figure
    fig.suptitle(fig_title, y=1.10)

    # Panel 1 (BLEU)
    h, lab = plotter(
        x_axis=x_axis, xlabel=x_label, ylabel=y1_label,
        title=None,  # no per-axis title
        separation_conventional=m1_sep, SPF=m1_spf, SLF=m1_slf,
        sentence_decode=m1_ssf, sentence_predict=None,
        AE_baseline=m1_ae, LLM_baseline=m1_llm,
        ax=axes[0], legend=False
    )

    # Panel 2 (SBERT)
    plotter(
        x_axis=x_axis, xlabel=x_label, ylabel=y2_label,
        title=None,
        separation_conventional=m2_sep, SPF=m2_spf, SLF=m2_slf,
        sentence_decode=m2_ssf, sentence_predict=None,
        AE_baseline=m2_ae, LLM_baseline=m2_llm,
        ax=axes[1], legend=False
    )

    # Panel 3 (GPT)
    plotter(
        x_axis=x_axis, xlabel=x_label, ylabel=y3_label,
        title=None,
        separation_conventional=m3_sep, SPF=m3_spf, SLF=m3_slf,
        sentence_decode=m3_ssf, sentence_predict=None,
        AE_baseline=m3_ae, LLM_baseline=m3_llm,
        ax=axes[2], legend=False
    )

    # Shared legend: single row
    fig.legend(
        h, lab,
        loc="upper center",
        ncol=len(lab),  # force one line
        frameon=True,
        fancybox=True,
        bbox_to_anchor=(0.5, 1.22),
        borderaxespad=0.2,
        columnspacing=1.2,
        handlelength=2.2,
        fontsize=11.2
    )

    # Leave space for legend + suptitle
    fig.tight_layout(rect=[0, 0, 1, 0.82])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")  # PDF inferred by extension
    if show:
        plt.show()
    plt.close(fig)


def _plot_three_metrics_figure_compact(
    x_axis,
    x_label,
    fig_title,
    # BLEU (1D)
    bleu_sep, bleu_spf, bleu_slf, bleu_ssf, bleu_ae, bleu_llm,
    # SBERT (1D)
    sbert_sep, sbert_spf, sbert_slf, sbert_ssf, sbert_ae, sbert_llm,
    # GPT (1D)
    gpt_sep, gpt_spf, gpt_slf, gpt_ssf, gpt_ae, gpt_llm,
    out_path,
    show=False,
):
    plt.style.use(["science", "ieee", "no-latex"])
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "STIXGeneral"],
        "mathtext.fontset": "stix",
    })

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.25))

    # Panels
    h, lab = plotter(
        x_axis=x_axis, xlabel=x_label, ylabel="BLEU Score", title=None,
        separation_conventional=bleu_sep, SPF=bleu_spf, SLF=bleu_slf,
        sentence_decode=bleu_ssf, sentence_predict=None,
        AE_baseline=bleu_ae, LLM_baseline=bleu_llm,
        ax=axes[0], legend=False
    )
    plotter(
        x_axis=x_axis, xlabel=x_label, ylabel="SBERT Score", title=None,
        separation_conventional=sbert_sep, SPF=sbert_spf, SLF=sbert_slf,
        sentence_decode=sbert_ssf, sentence_predict=None,
        AE_baseline=sbert_ae, LLM_baseline=sbert_llm,
        ax=axes[1], legend=False
    )
    plotter(
        x_axis=x_axis, xlabel=x_label, ylabel="GPT Score", title=None,
        separation_conventional=gpt_sep, SPF=gpt_spf, SLF=gpt_slf,
        sentence_decode=gpt_ssf, sentence_predict=None,
        AE_baseline=gpt_ae, LLM_baseline=gpt_llm,
        ax=axes[2], legend=False
    )

    # Compact global title + legend (INSIDE figure)
    # fig.suptitle(fig_title, y=0.999) # no title
    fig.legend(
        h, lab,
        loc="upper center",
        ncol=len(lab),            # one row
        frameon=True,
        fancybox=True,
        bbox_to_anchor=(0.5, 0.95),
        borderaxespad=0.0,
        columnspacing=0.9,
        handlelength=1.9,
        fontsize=11.2
    )

    # Compact spacing tuned for papers (no bbox_inches="tight")
    fig.subplots_adjust(
        left=0.06, right=0.995,
        bottom=0.18,
        top=0.85,
        wspace=0.18
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_three_metrics(
    d_sd_list,
    gamma_list,

    bleu_sep, bleu_spf, bleu_slf, bleu_ssf, bleu_ae, bleu_llm,
    sbert_sep, sbert_spf, sbert_slf, sbert_ssf, sbert_ae, sbert_llm,
    gpt_sep, gpt_spf, gpt_slf, gpt_ssf, gpt_ae, gpt_llm,

    save_dir="Plots",
    show=False,
):
    os.makedirs(save_dir, exist_ok=True)

    # A) sweep d_SR for each d_SD
    for i, d_sd in enumerate(d_sd_list):
        x_axis = np.array(gamma_list) * d_sd
        fig_title = rf"$d_{{SR}}$ v. Performance Scores for $d_{{SD}}={d_sd}$m"
        out_path = os.path.join(save_dir, f"three_metrics_dSR_dSD_{d_sd}m.pdf")

        _plot_three_metrics_figure_compact(
            x_axis=x_axis,
            x_label=r"$d_{SR}$ (m)",
            fig_title=fig_title,

            bleu_sep=bleu_sep[i, :], bleu_spf=bleu_spf[i, :], bleu_slf=bleu_slf[i, :],
            bleu_ssf=bleu_ssf[i, :], bleu_ae=bleu_ae[i, :],   bleu_llm=bleu_llm[i, :],

            sbert_sep=sbert_sep[i, :], sbert_spf=sbert_spf[i, :], sbert_slf=sbert_slf[i, :],
            sbert_ssf=sbert_ssf[i, :], sbert_ae=sbert_ae[i, :],   sbert_llm=sbert_llm[i, :],

            gpt_sep=gpt_sep[i, :], gpt_spf=gpt_spf[i, :], gpt_slf=gpt_slf[i, :],
            gpt_ssf=gpt_ssf[i, :], gpt_ae=gpt_ae[i, :],   gpt_llm=gpt_llm[i, :],

            out_path=out_path,
            show=show,
        )

    # B) sweep d_SD at gamma=0.5
    mid_index = gamma_list.index(0.5)
    x_axis = np.array(d_sd_list)
    fig_title = r"$d_{SD}$ v. Performance Scores for $d_{SR}=0.5d_{SD}$"
    out_path = os.path.join(save_dir, "three_metrics_dSD_gamma_0p5.pdf")

    _plot_three_metrics_figure_compact(
        x_axis=x_axis,
        x_label=r"$d_{SD}$ (m)",
        fig_title=fig_title,

        bleu_sep=bleu_sep[:, mid_index], bleu_spf=bleu_spf[:, mid_index], bleu_slf=bleu_slf[:, mid_index],
        bleu_ssf=bleu_ssf[:, mid_index], bleu_ae=bleu_ae[:, mid_index],   bleu_llm=bleu_llm[:, mid_index],

        sbert_sep=sbert_sep[:, mid_index], sbert_spf=sbert_spf[:, mid_index], sbert_slf=sbert_slf[:, mid_index],
        sbert_ssf=sbert_ssf[:, mid_index], sbert_ae=sbert_ae[:, mid_index],   sbert_llm=sbert_llm[:, mid_index],

        gpt_sep=gpt_sep[:, mid_index], gpt_spf=gpt_spf[:, mid_index], gpt_slf=gpt_slf[:, mid_index],
        gpt_ssf=gpt_ssf[:, mid_index], gpt_ae=gpt_ae[:, mid_index],   gpt_llm=gpt_llm[:, mid_index],

        out_path=out_path,
        show=show,
    )
