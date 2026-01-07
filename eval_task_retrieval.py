import argparse
import os
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.models.semantic_decoder import SemanticDecoder
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.models.semantic_transformer import (
    SemanticTransformer,
    ChannelEncoder,
    ChannelDecoder,
)
from semantic_communication.models.transceiver import Transceiver
from semantic_communication.utils.channel import init_channel
from semantic_communication.utils.general import (
    get_device,
    set_seed,
    add_semantic_decoder_args,
    add_channel_model_args,
    add_data_args,
    load_model,
)


def _decode_predicted_texts(
    predicted_ids: torch.Tensor,
    semantic_encoder: SemanticEncoder,
) -> List[str]:
    """
    Convert predicted token IDs into list of strings.
    Truncates each sequence at first [SEP] (id=2) if present.
    """
    # Find [SEP] per row; if none exists, argmax returns 0 -> treat as "no sep"
    sep_indices = torch.argmax((predicted_ids == 2).long(), dim=1)

    input_ids_list = []
    for i in range(predicted_ids.shape[0]):
        k = sep_indices[i].item()
        if k == 0:
            input_ids_list.append(predicted_ids[i, :])
        else:
            input_ids_list.append(predicted_ids[i, : k + 1])

    token_ids_list = [
        semantic_encoder.label_encoder.inverse_transform(ids) for ids in input_ids_list
    ]

    predicted_texts = semantic_encoder.get_tokens(
        token_ids=token_ids_list,
        skip_special_tokens=True,
    )
    # strip whitespace for cleaner matching
    return [t.strip() for t in predicted_texts]


def _decode_original_texts(
    encoder_idx: torch.Tensor,
    semantic_encoder: SemanticEncoder,
) -> List[str]:
    """
    Convert original input IDs into list of strings.
    """
    original_texts = semantic_encoder.get_tokens(
        ids=encoder_idx,
        skip_special_tokens=True,
    )
    return [t.strip() for t in original_texts]


def _dedup_by_original(
    originals: List[str], decoded: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Remove duplicate originals (exact string match) to avoid ambiguous retrieval.
    Keeps first occurrence.
    """
    seen = set()
    kept_orig = []
    kept_dec = []
    for o, d in zip(originals, decoded):
        if o in seen:
            continue
        seen.add(o)
        kept_orig.append(o)
        kept_dec.append(d)
    return kept_orig, kept_dec


@torch.no_grad()
def _compute_retrieval_metrics(
    originals: List[str],
    decoded: List[str],
    embedder: SentenceTransformer,
    k_list: List[int],
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Compute Recall@K and MRR.
    Returns:
      summary_metrics: {"recall@1": ..., "recall@5": ..., "mrr": ..., "mean_diag_cos": ...}
      per_example: {"top1": np.array, "rank": np.array}
    """
    assert len(originals) == len(decoded), "original/decoded length mismatch"
    n = len(originals)
    if n == 0:
        raise ValueError("No samples to score.")

    # Encode on embedder's device (often CPU for safety)
    orig_emb = embedder.encode(
        originals,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    dec_emb = embedder.encode(
        decoded,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Ensure both tensors on same device for matmul (keep on CPU by default)
    if orig_emb.device != dec_emb.device:
        dec_emb = dec_emb.to(orig_emb.device)

    sim = dec_emb @ orig_emb.T  # cosine similarity because normalized

    # True index is i
    true_idx = torch.arange(n, device=sim.device)

    # Top-1
    top1 = torch.argmax(sim, dim=1)
    recall1 = (top1 == true_idx).float().mean().item()

    # Recall@K for each K
    recall_at_k = {}
    for k in k_list:
        k_eff = min(k, n)
        topk = torch.topk(sim, k=k_eff, dim=1).indices
        hit = (topk == true_idx.unsqueeze(1)).any(dim=1).float().mean().item()
        recall_at_k[f"recall@{k}"] = hit

    # MRR: rank of the true item in sorted list
    # argsort descending: position 0 is best
    order = torch.argsort(sim, dim=1, descending=True)
    # Find where true index sits
    # (n, n) boolean is ok for n<=5000; your default is 500 so it's cheap
    match = order == true_idx.unsqueeze(1)
    ranks = torch.argmax(match.long(), dim=1)  # 0-based rank
    mrr = (1.0 / (ranks.float() + 1.0)).mean().item()

    # Mean diagonal cosine similarity is also nice to report
    mean_diag_cos = sim.diag().mean().item()

    summary = {
        "n": n,
        "recall@1": recall1,
        "mrr": mrr,
        "mean_diag_cos": mean_diag_cos,
    }
    summary.update(recall_at_k)

    per_ex = {
        "top1": top1.detach().cpu().numpy(),
        "rank": ranks.detach().cpu().numpy(),
    }
    return summary, per_ex


def main():
    parser = argparse.ArgumentParser()

    # model args
    parser.add_argument("--transceiver-path", type=str, required=True)

    add_semantic_decoder_args(parser)
    add_channel_model_args(parser)
    add_data_args(parser)

    # eval args
    parser.add_argument("--batch-size", default=125, type=int)
    parser.add_argument("--gamma-list", nargs="+", type=float, required=True)
    parser.add_argument("--d-list", nargs="+", type=float, required=True)
    parser.add_argument("--n-test", default=500, type=int)

    parser.add_argument("--results-dir", default="ResultsTask", type=str)
    parser.add_argument("--save-examples", action="store_true")
    parser.add_argument("--dedup-originals", action="store_true")

    # decoding choice
    parser.add_argument("--greedy", action="store_true")

    # retrieval embedder
    parser.add_argument(
        "--embedder",
        default="sentence-transformers/all-MiniLM-L6-v2",
        type=str,
        help="Sentence embedding model used ONLY for evaluation.",
    )
    parser.add_argument(
        "--embedder-device",
        default="cpu",
        type=str,
        help="cpu is safest (avoids GPU OOM with your transceiver). Use cuda if you know it fits.",
    )
    parser.add_argument(
        "--k-list",
        nargs="+",
        type=int,
        default=[1, 5],
        help="Compute Recall@K for these K values (e.g. --k-list 1 5 10).",
    )

    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    device = get_device()
    set_seed()

    # Data
    data_handler = DataHandler(
        batch_size=args.batch_size,
        data_fp=args.data_fp,
        mode=args.mode,
    )

    # Build models (must match checkpoint shapes!)
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
        state_memory_len=args.state_memory_len,  # IMPORTANT
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
        state_memory_len=args.state_memory_len,  # IMPORTANT
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

    # Evaluation embedder (CPU by default to avoid GPU OOM)
    embedder = SentenceTransformer(args.embedder, device=args.embedder_device)

    d_list = args.d_list
    gamma_list = args.gamma_list

    summary_rows = []
    example_rows = []

    # Arrays for easy RD plotting
    recall1_mat = np.zeros((len(d_list), len(gamma_list)), dtype=np.float32)
    mrr_mat = np.zeros((len(d_list), len(gamma_list)), dtype=np.float32)
    diagcos_mat = np.zeros((len(d_list), len(gamma_list)), dtype=np.float32)

    # Also store Recall@K (first non-1 K if provided)
    extra_k = [k for k in args.k_list if k != 1]
    recallk_mats = {
        k: np.zeros((len(d_list), len(gamma_list)), dtype=np.float32) for k in extra_k
    }

    for di, d_sd in enumerate(d_list):
        for gi, gamma in enumerate(gamma_list):
            d_sr = d_sd * gamma
            print(
                f"\n[Task Retrieval] d_sd={d_sd}  gamma={gamma}  d_sr={d_sr}  mem={args.state_memory_len}"
            )

            originals = []
            decoded = []

            # Collect N examples
            for b in tqdm(data_handler.test_dataloader, desc="Generating", leave=False):
                encoder_idx = b[0].to(device)
                encoder_attention_mask = b[1].to(device)
                encoder_idx = data_handler.label_encoder.transform(encoder_idx)

                out = transceiver.generate(
                    input_ids=encoder_idx,
                    attention_mask=encoder_attention_mask,
                    d_sd=float(d_sd),
                    d_sr=float(d_sr),
                    greedy=bool(args.greedy),
                )

                if isinstance(out, tuple):
                    predicted_ids = out[0]
                else:
                    predicted_ids = out

                pred_texts = _decode_predicted_texts(
                    predicted_ids, semantic_encoder=semantic_encoder
                )
                orig_texts = _decode_original_texts(
                    encoder_idx, semantic_encoder=semantic_encoder
                )

                originals.extend(orig_texts)
                decoded.extend(pred_texts)

                if len(originals) >= args.n_test:
                    break

            originals = originals[: args.n_test]
            decoded = decoded[: args.n_test]

            # Warn about duplicates (ambiguous retrieval)
            if len(set(originals)) < len(originals):
                print(
                    f"WARNING: {len(originals) - len(set(originals))} duplicate originals in this subset. "
                    f"Retrieval may be ambiguous. Consider --dedup-originals."
                )

            if args.dedup_originals:
                originals, decoded = _dedup_by_original(originals, decoded)
                print(f"After dedup: n={len(originals)}")

            # Compute retrieval metrics
            metrics, per_ex = _compute_retrieval_metrics(
                originals=originals,
                decoded=decoded,
                embedder=embedder,
                k_list=args.k_list,
                device=device,
            )

            recall1_mat[di, gi] = metrics["recall@1"]
            mrr_mat[di, gi] = metrics["mrr"]
            diagcos_mat[di, gi] = metrics["mean_diag_cos"]

            for k in extra_k:
                recallk_mats[k][di, gi] = metrics.get(f"recall@{k}", np.nan)

            summary_rows.append(
                {
                    "transceiver_path": args.transceiver_path,
                    "mode": args.mode,
                    "state_memory_len": args.state_memory_len,
                    "channel_type": args.channel_type,
                    "d_sd": float(d_sd),
                    "gamma": float(gamma),
                    "d_sr": float(d_sr),
                    **metrics,
                }
            )

            if args.save_examples:
                top1 = per_ex["top1"]
                rank = per_ex["rank"]
                for i, (o, d) in enumerate(zip(originals, decoded)):
                    example_rows.append(
                        {
                            "d_sd": float(d_sd),
                            "gamma": float(gamma),
                            "state_memory_len": args.state_memory_len,
                            "original": o,
                            "decoded": d,
                            "top1_correct": int(top1[i] == i),
                            "rank": int(rank[i]),
                        }
                    )

            # Save intermediate matrices after each condition (safe on clusters)
            np.save(
                os.path.join(args.results_dir, "retrieval_recall@1.npy"), recall1_mat
            )
            np.save(os.path.join(args.results_dir, "retrieval_mrr.npy"), mrr_mat)
            np.save(
                os.path.join(args.results_dir, "retrieval_mean_diag_cos.npy"),
                diagcos_mat,
            )
            for k, mat in recallk_mats.items():
                np.save(
                    os.path.join(args.results_dir, f"retrieval_recall@{k}.npy"), mat
                )

            # Also save/update summary CSV
            pd.DataFrame(summary_rows).to_csv(
                os.path.join(args.results_dir, "retrieval_summary.csv"),
                index=False,
            )

            if args.save_examples:
                pd.DataFrame(example_rows).to_csv(
                    os.path.join(args.results_dir, "retrieval_examples.csv"),
                    index=False,
                )

            print(
                f"Done: n={metrics['n']}  "
                f"R@1={metrics['recall@1']:.3f}  "
                + (
                    f"R@{extra_k[0]}={metrics.get(f'recall@{extra_k[0]}', np.nan):.3f}  "
                    if len(extra_k) > 0
                    else ""
                )
                + f"MRR={metrics['mrr']:.3f}  diagCos={metrics['mean_diag_cos']:.3f}"
            )


if __name__ == "__main__":
    main()
