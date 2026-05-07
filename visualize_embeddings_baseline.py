import argparse
import os
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from semantic_communication.models.baseline_models import Tx_Relay, Tx_Relay_Rx
from semantic_communication.models.semantic_transformer import SemanticTransformer
from semantic_communication.models.transceiver import (
    Transceiver,
    ChannelEncoder,
    ChannelDecoder,
)
import seaborn as sns
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import umap

from semantic_communication.utils.general import (
    get_device,
    set_seed,
    add_semantic_decoder_args,
    add_channel_model_args,
    add_data_args,
    load_model,
)
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.models.semantic_decoder import SemanticDecoder
from semantic_communication.utils.channel import init_channel
from semantic_communication.utils.eval_functions import *


def _get_predicted_tokens(predicted_ids):
    # find the end of sentences
    sep_indices = torch.argmax((predicted_ids == 2).long(), dim=1)
    input_ids_list = []
    for i in range(predicted_ids.shape[0]):
        k = sep_indices[i]
        if k == 0:  # no [SEP] predicted
            input_ids_list.append(predicted_ids[i, :])
        else:
            input_ids_list.append(predicted_ids[i, : k + 1])

    token_ids_list = [
        semantic_encoder.label_encoder.inverse_transform(input_ids)
        for input_ids in input_ids_list
    ]

    predicted_tokens = semantic_encoder.get_tokens(
        token_ids=token_ids_list,
        skip_special_tokens=True,
    )

    return predicted_tokens


def _mean_pool_embeddings(embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pool across sequence length ignoring padded positions."""
    mask = attention_mask.unsqueeze(-1).type_as(embeddings)
    denom = attention_mask.sum(dim=1, keepdim=True).clamp(min=1).type_as(embeddings)
    return (embeddings * mask).sum(dim=1) / denom


def _prepare_tsne_inputs(src_chunks, relay_chunks, dst_chunks):
    src = torch.cat(src_chunks, dim=0).cpu().numpy()
    relay = torch.cat(relay_chunks, dim=0).cpu().numpy()
    dst = torch.cat(dst_chunks, dim=0).cpu().numpy()
    return src, relay, dst, src.shape[0], relay.shape[0], dst.shape[0]


def _run_tsne_and_save(
    src_chunks,
    relay_chunks,
    dest_chunks,
    args,
    level: str,
    d_sd: float,
    gamma: float,
    results_dir: str,
):
    # --- Prepare embeddings ---
    src, relay, dst, n_src, n_relay, n_dst = _prepare_tsne_inputs(src_chunks, relay_chunks, dest_chunks)

    # Need aligned sentence-level triples for a trajectory plot
    if level != "sentence" or n_src == 0 or n_src != n_relay or n_dst != n_src:
        print(f"[UMAP] Skipping: need aligned sentence-level src/relay/dst. Got src={n_src}, relay={n_relay}, dst={n_dst}.")
        return

    # --- Choose 5 sentence indices ---
    rng = np.random.default_rng(int(getattr(args, "dr_seed", 0)))
    k = int(getattr(args, "dr_num_samples", 5))
    k = min(k, n_src)
    idxs = rng.choice(np.arange(n_src), size=k, replace=False)

    # --- Fit UMAP on ALL points, plot ONLY selected trajectories ---
    X = np.vstack([src, relay, dst])
    X = StandardScaler().fit_transform(X)

    Z = umap.UMAP(
        n_components=2,
        n_neighbors=int(getattr(args, "umap_n_neighbors", 15)),
        min_dist=float(getattr(args, "umap_min_dist", 0.1)),
        metric=str(getattr(args, "umap_metric", "euclidean")),
        random_state=0,
    ).fit_transform(X).astype(np.float32)

    Z_src = Z[:n_src]
    Z_relay = Z[n_src:n_src + n_relay]
    Z_dst = Z[n_src + n_relay:]

    # --- Exaggerate displacement so src/relay/dst don’t sit on top of each other ---
    # If args.dr_disp_scale <= 0, auto-scale so typical step length ~ dr_target_step
    disp_scale = float(getattr(args, "dr_disp_scale", 0.0))
    if disp_scale <= 0:
        target = float(getattr(args, "dr_target_step", 0.6))  # increase if still too small
        med = float(np.median(np.linalg.norm(Z_relay - Z_src, axis=1)))
        disp_scale = target / (med + 1e-9)
        disp_scale = min(disp_scale, 200.0)  # safety clamp

    # Build exaggerated points for selected indices
    S = Z_src[idxs]
    R = S + disp_scale * (Z_relay[idxs] - S)
    D = R + disp_scale * (Z_dst[idxs] - Z_relay[idxs])

    # --- Plot ---
    os.makedirs(results_dir, exist_ok=True)
    fig, ax = plt.subplots()

    # points (different shapes)
    ax.scatter(S[:, 0], S[:, 1], marker="o", label="src")
    ax.scatter(R[:, 0], R[:, 1], marker="x", label="relay")
    ax.scatter(D[:, 0], D[:, 1], marker="^", label="dst")

    # arrows (direction)
    for s, r, d in zip(S, R, D):
        ax.annotate("", xy=(r[0], r[1]), xytext=(s[0], s[1]),
                    arrowprops=dict(arrowstyle="->"))
        ax.annotate("", xy=(d[0], d[1]), xytext=(r[0], r[1]),
                    arrowprops=dict(arrowstyle="->"))

    ax.set_title(f"UMAP trajectories (k={k}, scale={disp_scale:.1f})  d_sd={d_sd}  gamma={gamma}")
    ax.legend()

    out = os.path.join(
        results_dir,
        f"{args.tsne_out_prefix}_umap_traj_k-{k}_dsd-{d_sd}_gamma-{gamma}_seed-{args.dr_seed}.pdf"
    )
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[UMAP] Saved: {out}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model args
    parser.add_argument("--tx-relay-path", type=str)
    parser.add_argument("--tx-relay-rx-path", type=str)

    add_semantic_decoder_args(parser)
    add_channel_model_args(parser)
    add_data_args(parser)

    # test args
    parser.add_argument("--batch-size", default=125, type=int)
    parser.add_argument("--gamma-list", nargs="+", type=float)
    parser.add_argument("--d-list", nargs="+", type=float)
    parser.add_argument("--n-test", default=500, type=int)
    parser.add_argument("--do-tsne", action="store_true")
    parser.add_argument("--tsne-level", choices=["sentence", "token"], default="sentence")
    parser.add_argument("--tsne-max-points", type=int, default=None)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-pca-dim", type=int, default=10000)
    parser.add_argument("--tsne-metric", type=str, default="cosine")
    parser.add_argument("--tsne-out-prefix", type=str, default="tsne")
    parser.add_argument("--tsne-only-first-pair", action="store_true")

    args = parser.parse_args()
    device = get_device()
    set_seed()

    if args.tsne_max_points is None:
        args.tsne_max_points = 2000 if args.tsne_level == "sentence" else 10000

    results_dir = "Results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    data_handler = DataHandler(
        batch_size=args.batch_size,
        data_fp=args.data_fp,
        mode=args.mode,
    )

    # initialize models
    semantic_encoder = SemanticEncoder(
        label_encoder=data_handler.label_encoder,
        max_length=args.max_length,
        mode=args.mode,
        rate=args.rate,
    ).to(device)

    channel = init_channel(args.channel_type, args.sig_pow, args.alpha, args.noise_pow)
    num_classes = data_handler.vocab_size

    tx_relay_model = Tx_Relay(
        nin=num_classes,
        n_emb=args.channel_block_input_dim,
        n_latent=args.channel_block_latent_dim,
        channel=channel,
        entire_network_train=1,
    ).to(device)
    load_model(tx_relay_model, args.tx_relay_path)

    tx_relay_rx_model = Tx_Relay_Rx(
        nin=num_classes,
        n_emb=args.channel_block_input_dim,
        n_latent=args.channel_block_latent_dim,
        channel=channel,
        tx_relay_model=tx_relay_model,
    ).to(device)
    load_model(tx_relay_rx_model, args.tx_relay_rx_path)

    sbert = SemanticEncoder(
        label_encoder=data_handler.label_encoder,
        max_length=args.max_length,
        mode="sentence",
        rate=1,
    ).to(device)

    n_d = len(args.d_list)
    n_gamma = len(args.gamma_list)

    records = []
    tsne_ran_once = False
    # for each d_sd
    for distance_index, d_sd in enumerate(args.d_list):
        # for each gamma in gamma list
        for gamma_index, gamma in enumerate(args.gamma_list):
            print(f"Simulating for distance: {d_sd}  - Gamma: {gamma}")

            sbert_semantic_sim_scores = []
            cosine_scores = []
            bleu1_scores = []
            bleu_scores = []

            d_sr = d_sd * gamma

            collect_tsne = args.do_tsne and (not args.tsne_only_first_pair or not tsne_ran_once)
            tsne_src_chunks = [] if collect_tsne else None
            tsne_relay_chunks = [] if collect_tsne else None
            tsne_dest_chunks = [] if collect_tsne else None
            tsne_pairs_collected = 0
            tsne_src_token_count = 0
            tsne_relay_token_count = 0
            samples_processed = 0

            with torch.no_grad():
                for b in data_handler.test_dataloader:
                    if samples_processed >= args.n_test:
                        break

                    encoder_idx = b[0].to(device)
                    encoder_attention_mask = b[1].to(device)

                    remaining = args.n_test - samples_processed
                    if encoder_idx.size(0) > remaining:
                        encoder_idx = encoder_idx[:remaining]
                        encoder_attention_mask = encoder_attention_mask[:remaining]

                    encoder_idx = data_handler.label_encoder.transform(encoder_idx)

                    B, T = encoder_idx.shape
                    with torch.no_grad():
                        dst_logits, relay_logits, _ = tx_relay_rx_model(
                            encoder_idx[:, 1:],
                            encoder_attention_mask[:, 1:],
                            d_sd,
                            d_sr,
                            d_sd-d_sr,
                        )

                    predicted_ids = (torch.argmax(dst_logits, dim=-1)).reshape(
                        B, args.max_length
                    )
                    relay_decoded = (torch.argmax(relay_logits, dim=-1)).reshape(
                        B, args.max_length
                    )

                    samples_processed += encoder_idx.size(0)
                    relay_tokens = _get_predicted_tokens(relay_decoded)
                    destination_tokens = _get_predicted_tokens(predicted_ids)

                    pooled_src = sbert(input_ids=encoder_idx, attention_mask=encoder_attention_mask).squeeze(1)
                    pooled_relay = sbert(messages=relay_tokens).squeeze(1)
                    pooled_dest = sbert(messages=destination_tokens).squeeze(1)

                    if collect_tsne:
                        if args.tsne_max_points is not None and tsne_pairs_collected >= args.tsne_max_points:
                            continue
                        if args.tsne_max_points is not None:
                            remaining = args.tsne_max_points - tsne_pairs_collected
                            if remaining <= 0:
                                continue
                            if pooled_src.size(0) > remaining:
                                idx = torch.randperm(pooled_src.size(0), device=pooled_src.device)[:remaining]
                                pooled_src = pooled_src.index_select(0, idx)
                                pooled_relay = pooled_relay.index_select(0, idx)
                        tsne_src_chunks.append(pooled_src.cpu())
                        tsne_relay_chunks.append(pooled_relay.cpu())
                        tsne_dest_chunks.append(pooled_dest.cpu())
                        tsne_pairs_collected += pooled_src.size(0)

            fp = f"Results/{Path(args.tx_relay_rx_path).stem}/"
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            torch.save(tsne_src_chunks, f"{fp}/src_chunks.pt")
            torch.save(tsne_relay_chunks, f"{fp}/relay_chunks.pt")
            torch.save(tsne_dest_chunks, f"{fp}/dest_chunks.pt")
