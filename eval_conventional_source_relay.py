import numpy as np
from openai import OpenAI
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import pandas as pd
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
import torch
import argparse
from sentence_transformers import SentenceTransformer
from semantic_communication.conventional_tools.conventional_source_relay import source_relay_p2p_com
from semantic_communication.utils.eval_functions import *
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model args
    parser.add_argument("--tx-relay-path", type=str)
    parser.add_argument("--API-KEY", type=str)  # API KEY

    add_semantic_decoder_args(parser)
    add_channel_model_args(parser)
    add_data_args(parser)

    # test args
    parser.add_argument("--batch-size", default=125, type=int)
    parser.add_argument("--d-list", nargs="+", type=float)
    parser.add_argument("--n-test", default=500, type=int)

    args = parser.parse_args()
    set_seed()

    results_dir = "Results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    client = OpenAI(api_key=args.API_KEY)
    sbert_eval_model = SentenceTransformer("all-MiniLM-L6-v2")

    data_handler = DataHandler(
        batch_size=args.batch_size,
        data_fp=args.data_fp,
    )

    semantic_encoder = SemanticEncoder(
        label_encoder=data_handler.label_encoder,
        max_length=args.max_length,
        mode=args.mode,
        rate=args.rate,
    )

    channel = init_channel(args.channel_type, args.sig_pow, args.alpha, args.noise_pow)
    num_classes = data_handler.vocab_size

    n_d = len(args.d_list)

    mean_semantic_sim = np.zeros((n_d, 1))
    mean_sbert_semantic_sim = np.zeros((n_d, 1))
    mean_bleu_1 = np.zeros((n_d, 1))
    mean_bleu = np.zeros((n_d, 1))

    std_semantic_sim = np.zeros((n_d, 1))
    std_sbert_semantic_sim = np.zeros((n_d, 1))
    std_bleu_1 = np.zeros((n_d, 1))
    std_bleu = np.zeros((n_d, 1))
    smoothing_function = SmoothingFunction().method1
    records = []

    source_relay_comm = source_relay_p2p_com(data_handler.vocab_size,256, channel_type=args.channel_type, alpha=args.alpha, noise_pow=args.noise_pow, channel_code=False)
    # source_relay_comm = source_relay_p2p_com(data_handler.vocab_size, 512, channel_code=True)


    # source_relay_comm_512_channel_coded.bit_error_rate_control(512, 1000000, [0, 3, 6, 9, 12, 15, 18, 21])
    # source_relay_comm.bit_error_rate_control(256, 1000000, [0, 3, 6, 9, 12, 15, 18, 21])
    # source_relay_comm.bit_error_rate_control(128, 1000000, [0, 3, 6, 9, 12, 15, 18, 21])
    # source_relay_comm.bit_error_rate_control(64, 1000000, [0, 3, 6, 9, 12, 15, 18, 21])
    # source_relay_comm.bit_error_rate_control(32, 1000000, [0, 3, 6, 9, 12, 15, 18, 21])
    # source_relay_comm.bit_error_rate_control(16, 1000000, [0, 3, 6, 9, 12, 15, 18, 21])
    # source_relay_comm.bit_error_rate_control(4, 1000000, [0, 3, 6, 9, 12, 15, 18, 21])

    # For each d_sr
    for distance_index, d_sr in enumerate(args.d_list):
        print(f"Simulating for distance: {d_sr}")

        sbert_semantic_sim_scores = []
        cosine_scores = []
        bleu1_scores = []
        bleu_scores = []

        # snr_db = 10 * np.log10(1 / (np.power(d_sr, args.alpha) * args.noise_pow))

        for b in data_handler.test_dataloader:
            encoder_idx = b[0]
            encoder_attention_mask = b[1]
            encoder_idx = data_handler.label_encoder.transform(encoder_idx).cpu().numpy()

            predicted_ids = source_relay_comm.communicate(np.squeeze(encoder_idx), d_sr)
            predicted_ids = torch.tensor(predicted_ids, device=get_device())
            token_ids_list = semantic_encoder.label_encoder.inverse_transform(predicted_ids)

            s1 = semantic_encoder.get_tokens(
                token_ids=token_ids_list,
                skip_special_tokens=True,
            )

            s2 = semantic_encoder.get_tokens(
                ids=encoder_idx,
                skip_special_tokens=True,
            )

            s1 = " ".join(s1[1:-1])
            s2 = " ".join(s2[1:-1])

            # sim_score = semantic_similarity_score(s1, s2, client)
            bleu_1_score = sentence_bleu(
                [word_tokenize(s1)],
                word_tokenize(s2),
                weights=[1, 0, 0, 0],
                smoothing_function=smoothing_function,
            )
            bleu_score = sentence_bleu(
                [word_tokenize(s1)],
                word_tokenize(s2),
                smoothing_function=smoothing_function,
            )
            sbert_sim_score = sbert_semantic_similarity_score(s1, s2, sbert_model=sbert_eval_model)

            # cosine_scores.append(sim_score)
            bleu1_scores.append(bleu_1_score)
            bleu_scores.append(bleu_score)
            sbert_semantic_sim_scores.append(sbert_sim_score)

            # records.append([d_sr, s1, s2, sim_score, bleu_1_score, bleu_score, sbert_sim_score])

            if len(bleu1_scores) >= args.n_test:
                break

        n_test_samples = len(bleu1_scores)
        # cosine_scores = [x for x in cosine_scores if not np.isnan(x)]

        # mean_semantic_sim[distance_index, 0] = np.mean(cosine_scores)
        mean_sbert_semantic_sim[distance_index, 0] = np.mean(sbert_semantic_sim_scores)
        mean_bleu_1[distance_index, 0] = np.mean(bleu1_scores)
        mean_bleu[distance_index, 0] = np.mean(bleu_scores)

        # std_semantic_sim[distance_index, 0] = np.std(
        #     cosine_scores, ddof=1
        # ) / np.sqrt(n_test_samples)
        std_bleu_1[distance_index, 0] = np.std(bleu1_scores, ddof=1) / np.sqrt(
            n_test_samples
        )
        std_bleu[distance_index, 0] = np.std(bleu_scores, ddof=1) / np.sqrt(
            n_test_samples
        )

        std_sbert_semantic_sim[distance_index, 0] = np.std(sbert_semantic_sim_scores, ddof=1) / np.sqrt(
            n_test_samples)

        np.save(os.path.join(results_dir, f"sr_classic_conventional_mean_semantic_sim_{args.channel_type}.npy"), mean_semantic_sim)
        np.save(os.path.join(results_dir, f"sr_classic_conventional_mean_sbert_semantic_sim_{args.channel_type}.npy"), mean_sbert_semantic_sim)
        np.save(os.path.join(results_dir, f"sr_classic_conventional_mean_bleu_1_{args.channel_type}.npy"), mean_bleu_1)
        np.save(os.path.join(results_dir, f"sr_classic_conventional_mean_bleu_{args.channel_type}.npy"), mean_bleu)

        np.save(os.path.join(results_dir, f"sr_classic_conventional_std_semantic_sim_{args.channel_type}.npy"), std_semantic_sim)
        np.save(os.path.join(results_dir, f"sr_classic_conventional_std_sbert_semantic_sim_{args.channel_type}.npy"), std_sbert_semantic_sim)
        np.save(os.path.join(results_dir, f"sr_classic_conventional_std_bleu_1_{args.channel_type}.npy"), std_bleu_1)
        np.save(os.path.join(results_dir, f"sr_classic_conventional_std_bleu_{args.channel_type}.npy"), std_bleu)

        # df = pd.DataFrame(
        #     records,
        #     columns=[
        #         "d_sr",
        #         "Sentence 1",
        #         "Sentence 2",
        #         "Semantic Similarity Score",
        #         "BLEU 1 Gram Score",
        #         "BLEU Score",
        #         "SBERT Semantic Score"
        #     ],
        # )
        # df.to_excel(os.path.join(results_dir, f"classic_baseline_output_{args.channel_type}.xlsx"), index=False)
