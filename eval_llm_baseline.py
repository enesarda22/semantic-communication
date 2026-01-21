import os
import argparse
import numpy as np
import pandas as pd
import torch
from openai import OpenAI
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from semantic_communication.utils.general import (
    get_device,
    set_seed,
    add_semantic_decoder_args,
    add_channel_model_args,
    add_data_args,
)
from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.utils.channel import init_channel
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.utils.eval_functions import (
    semantic_similarity_score,
    sbert_semantic_similarity_score,
)
from semantic_communication.models.llm_baseline import Tx_Relay_LLMSC, Tx_Relay_Rx_LLMSC


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--API-KEY", type=str, required=True)

    # LLM-SC baseline args
    parser.add_argument("--llm-model-name", type=str, default="gpt2")  # e.g., lmsys/vicuna-7b-v1.5
    parser.add_argument("--M", type=int, default=None, help="Modulation order (square QAM): 4,16,64,...")
    parser.add_argument("--symbols-per-token", type=int, default=None, help="Allowed channel uses per token (auto-select M).")
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--candidate-topk", type=int, default=256)

    # Large model loading
    parser.add_argument("--torch-dtype", type=str, default=None, help="e.g. float16, bfloat16")

    add_semantic_decoder_args(parser)
    add_channel_model_args(parser)
    add_data_args(parser)

    parser.add_argument("--batch-size", default=125, type=int)
    parser.add_argument("--gamma-list", nargs="+", type=float, required=True)
    parser.add_argument("--d-list", nargs="+", type=float, required=True)
    parser.add_argument("--n-test", default=500, type=int)

    parser.add_argument("--results-dir", type=str, default="Results")
    return parser.parse_args()


def _trim_by_mask(ids_2d: torch.Tensor, mask_2d: torch.Tensor):
    out = []
    lengths = mask_2d.long().sum(dim=1).tolist()
    for i, L in enumerate(lengths):
        L = int(L)
        out.append(ids_2d[i, :L] if L > 0 else ids_2d[i, :0])
    return out


if __name__ == "__main__":
    args = parse_args()
    device = get_device()
    set_seed()

    os.makedirs(args.results_dir, exist_ok=True)

    client = OpenAI(api_key=args.API_KEY)
    sbert_eval_model = SentenceTransformer("all-MiniLM-L6-v2")
    smoothing_function = SmoothingFunction().method1

    data_handler = DataHandler(
        batch_size=args.batch_size,
        data_fp=args.data_fp,
    )

    semantic_encoder = SemanticEncoder(
        label_encoder=data_handler.label_encoder,
        max_length=args.max_length,
        mode=args.mode,
        rate=args.rate,
    ).to(device)

    channel = init_channel(args.channel_type, args.sig_pow, args.alpha, args.noise_pow)

    if args.M is None and args.symbols_per_token is None:
        raise ValueError("Provide either --M or --symbols-per-token.")
    if args.M is not None and args.symbols_per_token is not None:
        print("Warning: both --M and --symbols-per-token provided; using --symbols-per-token (auto modulation).")

    tx_relay_model = Tx_Relay_LLMSC(
        model_name=args.llm_model_name,
        M=args.M,
        symbols_per_token=args.symbols_per_token,
        channel=channel,
        beam_width=args.beam_width,
        candidate_topk=args.candidate_topk,
        entire_network_train=1,
        device=str(device),
        torch_dtype=args.torch_dtype,
    )

    tx_relay_rx_model = Tx_Relay_Rx_LLMSC(
        model_name=args.llm_model_name,
        M=args.M,
        symbols_per_token=args.symbols_per_token,
        channel=channel,
        tx_relay_model=tx_relay_model,
        beam_width=args.beam_width,
        candidate_topk=args.candidate_topk,
        device=str(device),
        torch_dtype=args.torch_dtype,
    )

    tok = tx_relay_rx_model.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Log chosen modulation / symbols-per-token
    if hasattr(tx_relay_model, "M"):
        print(f"Using modulation M={tx_relay_model.M}")
    if getattr(tx_relay_model, "symbols_per_token_target", None) is not None:
        print(f"symbols_per_token target={tx_relay_model.symbols_per_token_target}, actual={tx_relay_model.symbols_per_token_actual}")

    n_d = len(args.d_list)
    n_gamma = len(args.gamma_list)

    mean_semantic_sim = np.zeros((n_d, n_gamma))
    mean_sbert_semantic_sim = np.zeros((n_d, n_gamma))
    mean_bleu_1 = np.zeros((n_d, n_gamma))
    mean_bleu = np.zeros((n_d, n_gamma))

    std_semantic_sim = np.zeros((n_d, n_gamma))
    std_sbert_semantic_sim = np.zeros((n_d, n_gamma))
    std_bleu_1 = np.zeros((n_d, n_gamma))
    std_bleu = np.zeros((n_d, n_gamma))

    records = []

    for distance_index, d_sd in enumerate(args.d_list):
        for gamma_index, gamma in enumerate(args.gamma_list):
            print(f"Simulating for distance: {d_sd}  - Gamma: {gamma}")

            sbert_scores = []
            cosine_scores = []
            bleu1_scores = []
            bleu_scores = []

            d_sr = d_sd * gamma
            d_rd = d_sd - d_sr

            tx_relay_rx_model.eval()

            for b in data_handler.test_dataloader:
                batch_ids_raw = b[0].to(device)
                batch_mask_raw = b[1].to(device)
                batch_ids = data_handler.label_encoder.transform(batch_ids_raw)

                original_sentences = semantic_encoder.get_tokens(
                    ids=batch_ids,
                    skip_special_tokens=True,
                )

                enc_llm = tok(
                    original_sentences,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=args.max_length,
                    add_special_tokens=False,  # IMPORTANT
                )

                x_llm = enc_llm["input_ids"].to(device)
                mask_llm = enc_llm["attention_mask"].to(device)
                with torch.no_grad():
                    logits, _ = tx_relay_rx_model(
                        x_llm,
                        mask_llm,
                        d_sd,
                        d_sr,
                        d_rd,
                    )
                    pred_ids = torch.argmax(logits, dim=-1)

                    pred_list = _trim_by_mask(pred_ids, mask_llm)
                    predicted_sentences = [
                        tok.decode(ids_i.detach().cpu(), skip_special_tokens=True)
                        for ids_i in pred_list
                    ]

                    for s1, s2 in zip(original_sentences, predicted_sentences):
                        sim_score = semantic_similarity_score(s1, s2, client)
                        bleu_1 = sentence_bleu(
                            [word_tokenize(s1)],
                            word_tokenize(s2),
                            weights=[1, 0, 0, 0],
                            smoothing_function=smoothing_function,
                        )
                        bleu = sentence_bleu(
                            [word_tokenize(s1)],
                            word_tokenize(s2),
                            smoothing_function=smoothing_function,
                        )
                        sbert = sbert_semantic_similarity_score(s1, s2, sbert_model=sbert_eval_model)

                        cosine_scores.append(sim_score)
                        bleu1_scores.append(bleu_1)
                        bleu_scores.append(bleu)
                        sbert_scores.append(sbert)

                        records.append([d_sd, gamma, s1, s2, sim_score, bleu_1, bleu, sbert])

                if len(bleu1_scores) >= args.n_test:
                    break

            n_test_samples = len(bleu1_scores)
            cosine_scores = [x for x in cosine_scores if not np.isnan(x)]

            mean_semantic_sim[distance_index, gamma_index] = np.mean(cosine_scores) if cosine_scores else np.nan
            mean_sbert_semantic_sim[distance_index, gamma_index] = np.mean(sbert_scores) if sbert_scores else np.nan
            mean_bleu_1[distance_index, gamma_index] = np.mean(bleu1_scores) if bleu1_scores else np.nan
            mean_bleu[distance_index, gamma_index] = np.mean(bleu_scores) if bleu_scores else np.nan

            std_semantic_sim[distance_index, gamma_index] = (np.std(cosine_scores, ddof=1) / np.sqrt(n_test_samples)) if n_test_samples > 1 and cosine_scores else np.nan
            std_sbert_semantic_sim[distance_index, gamma_index] = (np.std(sbert_scores, ddof=1) / np.sqrt(n_test_samples)) if n_test_samples > 1 and sbert_scores else np.nan
            std_bleu_1[distance_index, gamma_index] = (np.std(bleu1_scores, ddof=1) / np.sqrt(n_test_samples)) if n_test_samples > 1 and bleu1_scores else np.nan
            std_bleu[distance_index, gamma_index] = (np.std(bleu_scores, ddof=1) / np.sqrt(n_test_samples)) if n_test_samples > 1 and bleu_scores else np.nan

            np.save(os.path.join(args.results_dir, f"llmsc_mean_semantic_sim_{args.channel_type}.npy"), mean_semantic_sim)
            np.save(os.path.join(args.results_dir, f"llmsc_mean_sbert_semantic_sim_{args.channel_type}.npy"), mean_sbert_semantic_sim)
            np.save(os.path.join(args.results_dir, f"llmsc_mean_bleu_1_{args.channel_type}.npy"), mean_bleu_1)
            np.save(os.path.join(args.results_dir, f"llmsc_mean_bleu_{args.channel_type}.npy"), mean_bleu)

            np.save(os.path.join(args.results_dir, f"llmsc_std_semantic_sim_{args.channel_type}.npy"), std_semantic_sim)
            np.save(os.path.join(args.results_dir, f"llmsc_std_sbert_semantic_sim_{args.channel_type}.npy"), std_sbert_semantic_sim)
            np.save(os.path.join(args.results_dir, f"llmsc_std_bleu_1_{args.channel_type}.npy"), std_bleu_1)
            np.save(os.path.join(args.results_dir, f"llmsc_std_bleu_{args.channel_type}.npy"), std_bleu)

            df = pd.DataFrame(
                records,
                columns=[
                    "d_sd",
                    "Gamma",
                    "Sentence 1",
                    "Sentence 2",
                    "Semantic Similarity Score",
                    "BLEU 1 Gram Score",
                    "BLEU Score",
                    "SBERT Semantic Score",
                ],
            )
            df.to_excel(os.path.join(args.results_dir, f"llmsc_baseline_output_{args.channel_type}.xlsx"), index=False)

    print("Done. Results saved under:", args.results_dir)
