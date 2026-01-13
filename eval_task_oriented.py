"""
Task-oriented evaluation suite for semantic communication.

Generates decoded texts under channel conditions and computes:
- Retrieval: Recall@1, Recall@K, MRR, mean diagonal cosine
- NLI: entailment rate (+ contradiction rate; optional bidirectional entailment)
- NER: entity F1 (macro) between original and decoded
- Sentiment: label agreement between original and decoded

Designed to address reviewer concerns about "task-oriented" evaluation beyond BLEU/SBERT.
"""

import argparse
import os
import re
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from transformers import pipeline

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


# -----------------------------
# Helpers: decoding text
# -----------------------------
def decode_predicted_texts(predicted_ids: torch.Tensor, semantic_encoder: SemanticEncoder) -> List[str]:
    """
    Convert predicted token IDs into list of strings.
    Truncates at first [SEP] (id=2) if present.
    """
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
    texts = semantic_encoder.get_tokens(token_ids=token_ids_list, skip_special_tokens=True)
    return [t.strip() for t in texts]


def decode_original_texts(encoder_idx: torch.Tensor, semantic_encoder: SemanticEncoder) -> List[str]:
    texts = semantic_encoder.get_tokens(ids=encoder_idx, skip_special_tokens=True)
    return [t.strip() for t in texts]


def dedup_by_original(originals: List[str], decoded: List[str]) -> Tuple[List[str], List[str]]:
    """Remove duplicate originals (exact match) to avoid ambiguous retrieval."""
    seen = set()
    keep_o, keep_d = [], []
    for o, d in zip(originals, decoded):
        if o in seen:
            continue
        seen.add(o)
        keep_o.append(o)
        keep_d.append(d)
    return keep_o, keep_d


# -----------------------------
# Retrieval metrics
# -----------------------------
@torch.no_grad()
def compute_retrieval_metrics(
    originals: List[str],
    decoded: List[str],
    embedder: SentenceTransformer,
    k_list: List[int],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Returns:
      summary: dict with recall@k, mrr, mean_diag_cos, n
      per_ex: dict with top1 indices and ranks
    """
    assert len(originals) == len(decoded)
    n = len(originals)
    if n == 0:
        raise ValueError("No samples to score.")

    orig_emb = embedder.encode(
        originals, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False
    )
    dec_emb = embedder.encode(
        decoded, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False
    )
    if orig_emb.device != dec_emb.device:
        dec_emb = dec_emb.to(orig_emb.device)

    sim = dec_emb @ orig_emb.T  # cosine due to normalization
    true_idx = torch.arange(n, device=sim.device)

    top1 = torch.argmax(sim, dim=1)
    recall1 = (top1 == true_idx).float().mean().item()

    recall_at_k = {}
    for k in k_list:
        k_eff = min(int(k), n)
        topk = torch.topk(sim, k=k_eff, dim=1).indices
        hit = (topk == true_idx.unsqueeze(1)).any(dim=1).float().mean().item()
        recall_at_k[f"retrieval_recall@{k}"] = hit

    order = torch.argsort(sim, dim=1, descending=True)
    match = (order == true_idx.unsqueeze(1))
    ranks = torch.argmax(match.long(), dim=1)  # 0-based
    mrr = (1.0 / (ranks.float() + 1.0)).mean().item()

    mean_diag_cos = sim.diag().mean().item()

    summary = {"n": n, "retrieval_mrr": mrr, "retrieval_mean_diag_cos": mean_diag_cos}
    summary.update(recall_at_k)

    per_ex = {"retrieval_top1": top1.detach().cpu().numpy(),
              "retrieval_rank": ranks.detach().cpu().numpy()}
    return summary, per_ex


# -----------------------------
# NLI metrics
# -----------------------------
def hf_device_index(score_device: str) -> int:
    """Transformers pipeline device: -1 for cpu, 0/1/... for cuda."""
    s = score_device.lower()
    if s.startswith("cpu"):
        return -1
    if s.startswith("cuda"):
        if ":" in s:
            return int(s.split(":")[1])
        return 0
    # fallback
    return -1


def _label_is_entailment(label: str) -> bool:
    return "entail" in label.lower()


def _label_is_contradiction(label: str) -> bool:
    return "contrad" in label.lower()


@torch.no_grad()
def compute_nli_metrics(
    originals: List[str],
    decoded: List[str],
    nli_pipe,
    batch_size: int = 16,
    bidirectional: bool = False,
) -> Tuple[Dict[str, float], Optional[Dict[str, Any]]]:
    """
    NLI entailment rate (and contradiction rate).
    Direction: premise=original, hypothesis=decoded.

    If bidirectional=True, also compute entailment for reverse direction and
    bidirectional entailment rate (both directions entail).
    """
    assert len(originals) == len(decoded)
    n = len(originals)
    if n == 0:
        raise ValueError("No samples for NLI.")

    def run_pairs(prem: List[str], hyp: List[str]) -> List[str]:
        labels = []
        for i in range(0, n, batch_size):
            batch = [{"text": prem[j], "text_pair": hyp[j]} for j in range(i, min(i + batch_size, n))]
            out = nli_pipe(batch, truncation=True)
            # pipeline returns list of dicts
            labels.extend([o["label"] for o in out])
        return labels

    labels_fwd = run_pairs(originals, decoded)
    entail_fwd = np.mean([_label_is_entailment(l) for l in labels_fwd])
    contrad_fwd = np.mean([_label_is_contradiction(l) for l in labels_fwd])

    metrics = {
        "nli_entail_rate": float(entail_fwd),
        "nli_contrad_rate": float(contrad_fwd),
    }

    per_ex = {"nli_label_fwd": labels_fwd}

    if bidirectional:
        labels_rev = run_pairs(decoded, originals)
        entail_rev = np.mean([_label_is_entailment(l) for l in labels_rev])
        contrad_rev = np.mean([_label_is_contradiction(l) for l in labels_rev])
        bi_entail = np.mean(
            [(_label_is_entailment(a) and _label_is_entailment(b)) for a, b in zip(labels_fwd, labels_rev)]
        )

        metrics.update({
            "nli_entail_rate_rev": float(entail_rev),
            "nli_contrad_rate_rev": float(contrad_rev),
            "nli_bi_entail_rate": float(bi_entail),
        })
        per_ex["nli_label_rev"] = labels_rev

    return metrics, per_ex


# -----------------------------
# NER Entity F1
# -----------------------------
_entity_space_re = re.compile(r"\s+")
_entity_punct_re = re.compile(r"^[\W_]+|[\W_]+$")

def norm_entity(s: str) -> str:
    s = s.replace("##", "")
    s = _entity_space_re.sub(" ", s).strip().lower()
    s = _entity_punct_re.sub("", s)  # strip leading/trailing punctuation
    return s


@torch.no_grad()
def extract_entities(ner_pipe, texts: List[str], batch_size: int = 16) -> List[List[str]]:
    """
    Returns list per text: list of normalized entity strings.
    Uses HF token-classification pipeline with aggregation_strategy='simple'.
    """
    all_ents: List[List[str]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        out = ner_pipe(batch)
        # out: list of list[dict] if batch, or list[dict] if single
        if isinstance(out, dict) or (len(batch) == 1 and isinstance(out, list) and out and isinstance(out[0], dict)):
            out = [out]  # normalize to batch-of-texts

        for ents in out:
            if ents is None:
                all_ents.append([])
                continue
            # ents is list of dicts with 'word' and 'entity_group' etc
            normed = []
            for e in ents:
                w = norm_entity(e.get("word", ""))
                if w:
                    normed.append(w)
            all_ents.append(sorted(set(normed)))
    return all_ents


def entity_f1_from_sets(true_ents: List[str], pred_ents: List[str]) -> Tuple[float, float, float]:
    """Return (precision, recall, f1) for one example based on set overlap."""
    t = set(true_ents)
    p = set(pred_ents)

    if len(t) == 0 and len(p) == 0:
        return 1.0, 1.0, 1.0
    if len(t) == 0 and len(p) > 0:
        return 0.0, 0.0, 0.0
    if len(t) > 0 and len(p) == 0:
        return 0.0, 0.0, 0.0

    inter = len(t & p)
    prec = inter / max(len(p), 1)
    rec = inter / max(len(t), 1)
    if (prec + rec) == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return prec, rec, f1


@torch.no_grad()
def compute_entity_f1(
    originals: List[str],
    decoded: List[str],
    ner_pipe,
    batch_size: int = 16,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Macro entity precision/recall/F1 averaged over examples.
    """
    assert len(originals) == len(decoded)
    n = len(originals)
    if n == 0:
        raise ValueError("No samples for NER.")

    ents_o = extract_entities(ner_pipe, originals, batch_size=batch_size)
    ents_d = extract_entities(ner_pipe, decoded, batch_size=batch_size)

    precs, recs, f1s = [], [], []
    for eo, ed in zip(ents_o, ents_d):
        p, r, f = entity_f1_from_sets(eo, ed)
        precs.append(p); recs.append(r); f1s.append(f)

    metrics = {
        "entity_precision": float(np.mean(precs)),
        "entity_recall": float(np.mean(recs)),
        "entity_f1": float(np.mean(f1s)),
    }

    per_ex = {"entities_original": ents_o, "entities_decoded": ents_d, "entity_f1_per_ex": f1s}
    return metrics, per_ex


# -----------------------------
# Sentiment agreement
# -----------------------------
@torch.no_grad()
def compute_sentiment_agreement(
    originals: List[str],
    decoded: List[str],
    sent_pipe,
    batch_size: int = 32,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Predict sentiment on original and decoded and compute agreement rate.
    """
    assert len(originals) == len(decoded)
    n = len(originals)
    if n == 0:
        raise ValueError("No samples for sentiment.")

    def run(texts: List[str]) -> List[str]:
        labels = []
        for i in range(0, len(texts), batch_size):
            out = sent_pipe(texts[i:i + batch_size], truncation=True)
            labels.extend([o["label"] for o in out])
        return labels

    lab_o = run(originals)
    lab_d = run(decoded)

    agree = np.mean([a == b for a, b in zip(lab_o, lab_d)])

    metrics = {"sentiment_agreement": float(agree)}
    per_ex = {"sentiment_original": lab_o, "sentiment_decoded": lab_d}
    return metrics, per_ex


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--transceiver-path", type=str, required=True)

    add_semantic_decoder_args(parser)
    add_channel_model_args(parser)
    add_data_args(parser)

    # evaluation protocol
    parser.add_argument("--batch-size", default=125, type=int)
    parser.add_argument("--gamma-list", nargs="+", type=float, required=True)
    parser.add_argument("--d-list", nargs="+", type=float, required=True)
    parser.add_argument("--n-test", default=500, type=int)

    # what to compute
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["retrieval", "nli", "ner", "sentiment"],
        help="Any subset of: retrieval nli ner sentiment",
    )

    # output
    parser.add_argument("--results-dir", default="ResultsTask", type=str)
    parser.add_argument("--save-examples", action="store_true")
    parser.add_argument("--dedup-originals", action="store_true")

    # decoding
    parser.add_argument("--greedy", action="store_true")

    # retrieval embedder
    parser.add_argument("--embedder", default="all-MiniLM-L6-v2", type=str)
    parser.add_argument("--embedder-device", default="cpu", type=str)
    parser.add_argument("--k-list", nargs="+", type=int, default=[1, 5])

    # scoring models (HF)
    parser.add_argument("--score-device", default="cpu", type=str, help="cpu or cuda[:id]")
    parser.add_argument("--hf-batch-size", default=16, type=int)

    parser.add_argument("--nli-model", default="roberta-large-mnli", type=str)
    parser.add_argument("--nli-bidirectional", action="store_true")

    parser.add_argument("--ner-model", default="dslim/bert-base-NER", type=str)

    parser.add_argument("--sentiment-model", default="distilbert-base-uncased-finetuned-sst-2-english", type=str)

    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    set_seed()
    device = get_device()

    metrics_set = set([m.lower() for m in args.metrics])

    # Data
    data_handler = DataHandler(
        batch_size=args.batch_size,
        data_fp=args.data_fp,
        mode=args.mode,
    )

    # Build transceiver architecture
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

    # Retrieval embedder
    embedder = None
    if "retrieval" in metrics_set:
        embedder = SentenceTransformer(args.embedder, device=args.embedder_device)

    # HF pipelines
    hf_dev = hf_device_index(args.score_device)
    nli_pipe = None
    ner_pipe = None
    sent_pipe = None

    if "nli" in metrics_set:
        nli_pipe = pipeline(
            "text-classification",
            model=args.nli_model,
            device=hf_dev,
        )
    if "ner" in metrics_set:
        ner_pipe = pipeline(
            "token-classification",
            model=args.ner_model,
            device=hf_dev,
            aggregation_strategy="simple",
        )
    if "sentiment" in metrics_set:
        sent_pipe = pipeline(
            "sentiment-analysis",
            model=args.sentiment_model,
            device=hf_dev,
        )

    d_list = args.d_list
    gamma_list = args.gamma_list

    summary_rows: List[Dict[str, Any]] = []
    example_rows: List[Dict[str, Any]] = []

    # metric matrices for plotting
    def mat():
        return np.zeros((len(d_list), len(gamma_list)), dtype=np.float32)

    mats: Dict[str, np.ndarray] = {}
    # retrieval mats (if enabled)
    if "retrieval" in metrics_set:
        for k in args.k_list:
            mats[f"retrieval_recall@{k}"] = mat()
        mats["retrieval_mrr"] = mat()
        mats["retrieval_mean_diag_cos"] = mat()

    if "nli" in metrics_set:
        mats["nli_entail_rate"] = mat()
        mats["nli_contrad_rate"] = mat()
        if args.nli_bidirectional:
            mats["nli_bi_entail_rate"] = mat()

    if "ner" in metrics_set:
        mats["entity_f1"] = mat()
        mats["entity_precision"] = mat()
        mats["entity_recall"] = mat()

    if "sentiment" in metrics_set:
        mats["sentiment_agreement"] = mat()

    for di, d_sd in enumerate(d_list):
        for gi, gamma in enumerate(gamma_list):
            d_sr = float(d_sd) * float(gamma)
            print(f"\n=== Evaluating d_sd={d_sd} gamma={gamma} (d_sr={d_sr})")

            originals: List[str] = []
            decoded: List[str] = []

            # Generate outputs
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
                predicted_ids = out[0] if isinstance(out, tuple) else out

                decoded_batch = decode_predicted_texts(predicted_ids, semantic_encoder=semantic_encoder)
                original_batch = decode_original_texts(encoder_idx, semantic_encoder=semantic_encoder)

                originals.extend(original_batch)
                decoded.extend(decoded_batch)

                if len(originals) >= args.n_test:
                    break

            originals = originals[: args.n_test]
            decoded = decoded[: args.n_test]

            if args.dedup_originals:
                originals, decoded = dedup_by_original(originals, decoded)

            row: Dict[str, Any] = {
                "transceiver_path": args.transceiver_path,
                "mode": args.mode,
                "channel_type": args.channel_type,
                "d_sd": float(d_sd),
                "gamma": float(gamma),
                "d_sr": float(d_sr),
                "n": len(originals),
            }

            per_ex: Dict[str, Any] = {}

            # Compute metrics
            if "retrieval" in metrics_set:
                rmet, rper = compute_retrieval_metrics(originals, decoded, embedder, args.k_list)
                row.update(rmet)
                per_ex.update(rper)
                for k in args.k_list:
                    mats[f"retrieval_recall@{k}"][di, gi] = row[f"retrieval_recall@{k}"]
                mats["retrieval_mrr"][di, gi] = row["retrieval_mrr"]
                mats["retrieval_mean_diag_cos"][di, gi] = row["retrieval_mean_diag_cos"]

            if "nli" in metrics_set:
                nmet, nper = compute_nli_metrics(
                    originals, decoded, nli_pipe,
                    batch_size=args.hf_batch_size,
                    bidirectional=args.nli_bidirectional,
                )
                row.update(nmet)
                per_ex.update(nper)
                mats["nli_entail_rate"][di, gi] = row["nli_entail_rate"]
                mats["nli_contrad_rate"][di, gi] = row["nli_contrad_rate"]
                if args.nli_bidirectional and "nli_bi_entail_rate" in row:
                    mats["nli_bi_entail_rate"][di, gi] = row["nli_bi_entail_rate"]

            if "ner" in metrics_set:
                emet, eper = compute_entity_f1(originals, decoded, ner_pipe, batch_size=args.hf_batch_size)
                row.update(emet)
                per_ex.update(eper)
                mats["entity_f1"][di, gi] = row["entity_f1"]
                mats["entity_precision"][di, gi] = row["entity_precision"]
                mats["entity_recall"][di, gi] = row["entity_recall"]

            if "sentiment" in metrics_set:
                smet, sper = compute_sentiment_agreement(
                    originals, decoded, sent_pipe, batch_size=max(16, args.hf_batch_size * 2)
                )
                row.update(smet)
                per_ex.update(sper)
                mats["sentiment_agreement"][di, gi] = row["sentiment_agreement"]

            summary_rows.append(row)

            # Save per-example (optional)
            if args.save_examples:
                # keep it light: only save first N examples per condition
                for i in range(len(originals)):
                    ex = {
                        "d_sd": float(d_sd),
                        "gamma": float(gamma),
                        "state_memory_len": args.state_memory_len,
                        "original": originals[i],
                        "decoded": decoded[i],
                    }
                    if "retrieval_top1" in per_ex:
                        ex["retrieval_top1_correct"] = int(per_ex["retrieval_top1"][i] == i)
                        ex["retrieval_rank"] = int(per_ex["retrieval_rank"][i])
                    if "nli_label_fwd" in per_ex:
                        ex["nli_label_fwd"] = per_ex["nli_label_fwd"][i]
                    if "entities_original" in per_ex:
                        ex["entities_original"] = "; ".join(per_ex["entities_original"][i])
                        ex["entities_decoded"] = "; ".join(per_ex["entities_decoded"][i])
                        ex["entity_f1"] = float(per_ex["entity_f1_per_ex"][i])
                    if "sentiment_original" in per_ex:
                        ex["sentiment_original"] = per_ex["sentiment_original"][i]
                        ex["sentiment_decoded"] = per_ex["sentiment_decoded"][i]
                        ex["sentiment_agree"] = int(per_ex["sentiment_original"][i] == per_ex["sentiment_decoded"][i])
                    example_rows.append(ex)

            # Persist after each condition (cluster-safe)
            pd.DataFrame(summary_rows).to_csv(
                os.path.join(args.results_dir, "task_summary.csv"), index=False
            )
            if args.save_examples:
                pd.DataFrame(example_rows).to_csv(
                    os.path.join(args.results_dir, "task_examples.csv"), index=False
                )

            # Save metric matrices
            for name, M in mats.items():
                np.save(os.path.join(args.results_dir, f"{name}.npy"), M)

            # Print a compact line
            msg = f"Done n={row['n']} | "
            if "retrieval" in metrics_set:
                msg += f"R@1={row.get('retrieval_recall@1', np.nan):.3f} MRR={row.get('retrieval_mrr', np.nan):.3f} | "
            if "nli" in metrics_set:
                msg += f"NLI-ent={row.get('nli_entail_rate', np.nan):.3f} NLI-contr={row.get('nli_contrad_rate', np.nan):.3f} | "
            if "ner" in metrics_set:
                msg += f"EntF1={row.get('entity_f1', np.nan):.3f} | "
            if "sentiment" in metrics_set:
                msg += f"SentAgree={row.get('sentiment_agreement', np.nan):.3f}"
            print(msg)

    print("\nAll done. Outputs written to:", args.results_dir)


if __name__ == "__main__":
    main()
