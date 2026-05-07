import argparse
import os
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
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
    parser.add_argument("--transceiver-path", type=str)

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

    semantic_decoder = SemanticDecoder(
        vocab_size=data_handler.vocab_size,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        n_embeddings=args.n_embeddings,
        block_size=args.max_length,
        bert=semantic_encoder.bert,
        pad_idx=data_handler.label_encoder.pad_id,
    ).to(device)

    channel_encoder = ChannelEncoder(
        nin=args.channel_block_input_dim,
        nout=args.channel_block_latent_dim,
    ).to(device)

    channel_decoder = ChannelDecoder(
        nin=args.channel_block_latent_dim,
        nout=args.channel_block_input_dim,
    ).to(device)

    channel = init_channel(args.channel_type, args.sig_pow, args.alpha, args.noise_pow)

    semantic_transformer = SemanticTransformer(
        semantic_encoder=semantic_encoder,
        semantic_decoder=semantic_decoder,
        channel_encoder=channel_encoder,
        channel_decoder=channel_decoder,
        channel=channel,
    ).to(device)

    relay_semantic_encoder = SemanticEncoder(
        label_encoder=data_handler.label_encoder,
        max_length=args.max_length,
        mode=args.mode if args.mode == "sentence" else "forward",
        rate=1 if args.mode == "sentence" else None,
    ).to(device)

    relay_channel_encoder = ChannelEncoder(
        nin=args.channel_block_input_dim,
        nout=args.channel_block_latent_dim,
    ).to(device)

    dst_channel_decoder = ChannelDecoder(
        nin=args.channel_block_latent_dim * 2,
        nout=args.channel_block_input_dim,
    ).to(device)

    dst_semantic_decoder = SemanticDecoder(
        vocab_size=data_handler.vocab_size,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        n_embeddings=args.n_embeddings,
        block_size=args.max_length,
        bert=relay_semantic_encoder.bert,
        pad_idx=data_handler.label_encoder.pad_id,
    ).to(device)

    transceiver = Transceiver(
        src_relay_transformer=semantic_transformer,
        relay_semantic_encoder=relay_semantic_encoder,
        relay_channel_encoder=relay_channel_encoder,
        dst_channel_decoder=dst_channel_decoder,
        dst_semantic_decoder=dst_semantic_decoder,
        channel=channel,
        max_length=args.max_length,
    ).to(device)
    load_model(transceiver, args.transceiver_path)

    transceiver.eval()

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

                    predicted_ids, probs, relay_decoded = transceiver.generate(
                        input_ids=encoder_idx,
                        attention_mask=encoder_attention_mask,
                        d_sd=d_sd,
                        d_sr=d_sr,
                    )
                    samples_processed += encoder_idx.size(0)

                    # create tril attention mask
                    B, T = relay_decoded.shape
                    x_padding_mask = encoder_attention_mask[:, 1:] == 0
                    repeat_amounts = (~x_padding_mask).sum(dim=1)

                    relay_attention_mask = torch.tril(
                        torch.ones(T, T, device=device, dtype=torch.int64),
                        diagonal=1
                    )
                    relay_attention_mask = torch.cat(
                        [relay_attention_mask[:i, :] for i in repeat_amounts]
                    )

                    # re-encode decoded sentences and forward
                    relay_decoded = torch.repeat_interleave(relay_decoded, repeat_amounts, dim=0)
                    pooled_relay = sbert(input_ids=relay_decoded, attention_mask=relay_attention_mask).squeeze(1)

                    relay_chunks = torch.split(pooled_relay.cpu(), repeat_amounts.tolist(), dim=0)
                    tsne_relay_chunks.extend(relay_chunks)
                    tsne_pairs_collected += len(relay_chunks)

                    pooled_src = sbert(input_ids=encoder_idx, attention_mask=encoder_attention_mask).squeeze(1)
                    tsne_src_chunks.extend([pooled_src[i:i+1] for i in range(pooled_src.size(0))])


            fp = f"Results/{Path(args.transceiver_path).stem}/"
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            torch.save(tsne_relay_chunks, f"{fp}/relay_chunks.pt")
            torch.save(tsne_src_chunks, f"{fp}/src_chunks.pt")


# def plot_compare_two_folders_trajectories(
#     folder_a: str,
#     folder_b: str,
#     label_a: str,
#     label_b: str,
#     out_pdf: str,
#     k: int = 5,
#     seed: int = 0,
#     dr_method: str = "umap",          # "umap" | "pca" | "tsne"
#     umap_n_neighbors: int = 15,
#     umap_min_dist: float = 0.1,
#     umap_metric: str = "euclidean",
#     tsne_perplexity: float = 30.0,
#     disp_scale: float = 0.0,          # 0 => auto scale so arrows are visible
#     target_step: float = 0.6,         # used only when disp_scale==0
# ):
#     import os
#     import numpy as np
#     import torch
#     import matplotlib.pyplot as plt
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.decomposition import PCA
#     from sklearn.manifold import TSNE
#     from matplotlib.lines import Line2D
#
#     def _load_triplet(folder):
#         src_chunks = torch.load(os.path.join(folder, "src_chunks.pt"), map_location="cpu")
#         relay_chunks = torch.load(os.path.join(folder, "relay_chunks.pt"), map_location="cpu")
#         dest_chunks = torch.load(os.path.join(folder, "dest_chunks.pt"), map_location="cpu")
#         # Reuse your existing helper (recommended)
#         src, relay, dst, n_src, n_relay, n_dst = _prepare_tsne_inputs(src_chunks, relay_chunks, dest_chunks)
#         if n_src == 0 or n_src != n_relay or n_src != n_dst:
#             raise ValueError(f"Need aligned sentence-level triplets in {folder}. Got src={n_src}, relay={n_relay}, dst={n_dst}.")
#         return src, relay, dst, n_src
#
#     rng = np.random.default_rng(seed)
#
#     # --- Load both runs ---
#     srcA, relA, dstA, nA = _load_triplet(folder_a)
#     srcB, relB, dstB, nB = _load_triplet(folder_b)
#
#     n = min(nA, nB)  # common length
#     srcA, relA, dstA = srcA[:n], relA[:n], dstA[:n]
#     srcB, relB, dstB = srcB[:n], relB[:n], dstB[:n]
#
#     k = int(min(k, n))
#     idxs = rng.choice(np.arange(n), size=k, replace=False)
#
#     # --- Fit one shared 2D embedding on ALL points (both folders) ---
#     X = np.vstack([srcA, relA, dstA, srcB, relB, dstB])
#     X = StandardScaler().fit_transform(X)
#
#     dr_method = dr_method.lower()
#     if dr_method == "pca":
#         Z = PCA(n_components=2, random_state=0).fit_transform(X)
#
#     elif dr_method == "tsne":
#         N = X.shape[0]
#         perp = float(tsne_perplexity)
#         perp = min(perp, max(2.0, (N - 1) / 3.0))
#         perp = min(perp, N - 1)
#         Z = TSNE(
#             n_components=2,
#             perplexity=perp,
#             init="pca",
#             learning_rate="auto",
#             random_state=0,
#             max_iter=1000,
#         ).fit_transform(X).astype(np.float32)
#
#     elif dr_method == "umap":
#         try:
#             import umap
#         except ImportError as e:
#             raise ImportError("UMAP not installed. Install with: pip install umap-learn") from e
#         Z = umap.UMAP(
#             n_components=2,
#             n_neighbors=int(umap_n_neighbors),
#             min_dist=float(umap_min_dist),
#             metric=str(umap_metric),
#             random_state=0,
#         ).fit_transform(X).astype(np.float32)
#     else:
#         raise ValueError("dr_method must be one of: 'umap', 'pca', 'tsne'")
#
#     # split back
#     off = 0
#     ZsA = Z[off:off+n]; off += n
#     ZrA = Z[off:off+n]; off += n
#     ZdA = Z[off:off+n]; off += n
#     ZsB = Z[off:off+n]; off += n
#     ZrB = Z[off:off+n]; off += n
#     ZdB = Z[off:off+n]; off += n
#
#     # --- auto scale displacement if requested (to avoid "everything overlaps") ---
#     if disp_scale <= 0:
#         med = float(np.median(np.linalg.norm(ZrA - ZsA, axis=1)))
#         disp_scale = target_step / (med + 1e-9)
#         disp_scale = min(disp_scale, 200.0)
#
#     # --- Build the plotted coordinates (anchor both methods at same src per sentence) ---
#     S = ZsA[idxs]  # reference src locations (shared start)
#     RA = ZrA[idxs]
#     DA = ZdA[idxs]
#     SB = ZsB[idxs]
#     RB = ZrB[idxs]
#     DB = ZdB[idxs]
#
#     # shift method-B per sentence so its src matches method-A src
#     shift = (S - SB)
#     RB = RB + shift
#     DB = DB + shift
#
#     # exaggerate displacements so paths are visible
#     RA_plot = S + disp_scale * (RA - S)
#     DA_plot = RA_plot + disp_scale * (DA - RA)
#
#     RB_plot = S + disp_scale * (RB - S)
#     DB_plot = RB_plot + disp_scale * (DB - RB)
#
#     # --- Plot ---
#     fig, ax = plt.subplots()
#
#     colorA = "tab:blue"
#     colorB = "tab:orange"
#
#     # positions as shapes; method shown by color
#     ax.scatter(S[:, 0], S[:, 1], marker="o", c="k", s=40)
#     ax.scatter(RA_plot[:, 0], RA_plot[:, 1], marker="x", c=colorA, s=50)
#     ax.scatter(DA_plot[:, 0], DA_plot[:, 1], marker="^", c=colorA, s=50)
#     ax.scatter(RB_plot[:, 0], RB_plot[:, 1], marker="x", c=colorB, s=50)
#     ax.scatter(DB_plot[:, 0], DB_plot[:, 1], marker="^", c=colorB, s=50)
#
#     # arrows: src->relay->dst for both methods
#     for s, ra, da, rb, db in zip(S, RA_plot, DA_plot, RB_plot, DB_plot):
#         ax.annotate("", xy=(ra[0], ra[1]), xytext=(s[0], s[1]),
#                     arrowprops=dict(arrowstyle="->", color=colorA))
#         ax.annotate("", xy=(da[0], da[1]), xytext=(ra[0], ra[1]),
#                     arrowprops=dict(arrowstyle="->", color=colorA))
#
#         ax.annotate("", xy=(rb[0], rb[1]), xytext=(s[0], s[1]),
#                     arrowprops=dict(arrowstyle="->", color=colorB))
#         ax.annotate("", xy=(db[0], db[1]), xytext=(rb[0], rb[1]),
#                     arrowprops=dict(arrowstyle="->", color=colorB))
#
#     # legend: shapes = position, colors = method
#     handles = [
#         Line2D([0], [0], marker="o", color="k", linestyle="None", label="src"),
#         Line2D([0], [0], marker="x", color="k", linestyle="None", label="relay"),
#         Line2D([0], [0], marker="^", color="k", linestyle="None", label="dst"),
#         Line2D([0], [0], color=colorA, lw=2, label=label_a),
#         Line2D([0], [0], color=colorB, lw=2, label=label_b),
#     ]
#     ax.legend(handles=handles)
#
#     ax.set_title(f"{dr_method.upper()}")
#     os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
#     fig.savefig(out_pdf, bbox_inches="tight")
#     plt.close(fig)
#
#     print(f"Saved: {out_pdf}")
#     print(f"Indices used: {idxs.tolist()}")
# plot_compare_two_folders_trajectories(
#     folder_a="Results/transceiver_forward_AWGN_3",
#     folder_b="Results/transceiver_sentence_AWGN_14",
#     out_pdf="Results/compare_forward_vs_sentence_umap.pdf",
#     k=3,
#     seed=22,
#     dr_method="umap",
#     disp_scale=1.0,      # auto
#     target_step=0.6,
# )