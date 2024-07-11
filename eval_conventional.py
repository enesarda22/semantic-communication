import numpy as np
from openai import OpenAI
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import pandas as pd
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.utils.general import (
    get_device,
    set_seed,
    add_semantic_decoder_args,
    add_channel_model_args,
    add_data_args,
    split_string_by_lengths
)
from semantic_communication.conventional_tools.conventional_three_node_network import conventional_three_node_network
from semantic_communication.data_processing.data_handler import DataHandler
import argparse
from sentence_transformers import SentenceTransformer
from semantic_communication.utils.eval_functions import *
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model args
    parser.add_argument("--tx-relay-path", type=str)
    parser.add_argument("--tx-relay-rx-path", type=str)
    parser.add_argument("--API-KEY", type=str)  # API KEY

    add_semantic_decoder_args(parser)
    add_channel_model_args(parser)
    add_data_args(parser)

    # test args
    parser.add_argument("--batch-size", default=125, type=int)
    parser.add_argument("--gamma-list", nargs="+", type=float)
    parser.add_argument("--d-list", nargs="+", type=float)
    parser.add_argument("--d-grid", nargs="+", type=float)
    parser.add_argument("--n-test", default=500, type=int)

    args = parser.parse_args()
    device = get_device()
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
    ).to(device)

    conventional_three_node_network = conventional_three_node_network(data_handler=data_handler, channel_coding=False
                                                                      , channel_type=args.channel_type, sig_pow=args.sig_pow,
                                                                      alpha=args.alpha, noise_pow=args.noise_pow,
                                                                      d_grid=args.d_grid, train_transition=False,
                                                                      n_train=1000, data_fp="Data")

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
    smoothing_function = SmoothingFunction().method1
    records = []

    # For each d_sd
    for distance_index, d_sd in enumerate(args.d_list):
        # For each gamma in gamma list
        for gamma_index, gamma in enumerate(args.gamma_list):
            print(f"Simulating for distance: {d_sd}  - Gamma: {gamma}")

            sbert_semantic_sim_scores = []
            gpt_scores = []
            bleu1_scores = []
            bleu_scores = []

            d_sr = d_sd * gamma
            d_rd = d_sd - d_sr

            for b in data_handler.test_dataloader:
                encoder_idx = b[0]
                mask = (encoder_idx != 0) & (encoder_idx != 101) & (encoder_idx != 102)
                encoder_idx = encoder_idx.masked_select(mask)
                encoder_idx = data_handler.label_encoder.transform(encoder_idx).cpu().detach().numpy()
                valid_counts = mask.sum(dim=1).tolist()

                source_decoded = conventional_three_node_network(encoder_idx, d_sd, d_sr, d_rd, valid_counts)

                token_ids = semantic_encoder.label_encoder.inverse_transform(source_decoded)
                predicted_sentence = ' '.join(semantic_encoder.get_tokens(
                    token_ids=token_ids,
                    skip_special_tokens=True,
                ))

                original_sentence = ' '.join(semantic_encoder.get_tokens(
                    ids=encoder_idx,
                    skip_special_tokens=True,
                ))

                predicted_sentence = split_string_by_lengths(predicted_sentence, valid_counts)
                original_sentence = split_string_by_lengths(original_sentence, valid_counts)

                for s1, s2 in zip(original_sentence, predicted_sentence):
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

                    sbert_sim_score = sbert_semantic_similarity_score(
                        s1, s2, sbert_model=sbert_eval_model
                    )

                    # gpt_scores.append(sim_score)
                    bleu1_scores.append(bleu_1_score)
                    bleu_scores.append(bleu_score)
                    sbert_semantic_sim_scores.append(sbert_sim_score)

                    # records.append(
                    #     [
                    #         d_sd,
                    #         gamma,
                    #         s1,
                    #         s2,
                    #         sim_score,
                    #         bleu_1_score,
                    #         bleu_score,
                    #         sbert_sim_score,
                    #     ]
                    # )

                if len(bleu1_scores) >= args.n_test:
                    break

            n_test_samples = len(bleu1_scores)
            # gpt_scores = [x for x in gpt_scores if not np.isnan(x)]

            # mean_semantic_sim[distance_index, gamma_index] = np.mean(gpt_scores)
            mean_sbert_semantic_sim[distance_index, gamma_index] = np.mean(
                sbert_semantic_sim_scores
            )
            mean_bleu_1[distance_index, gamma_index] = np.mean(bleu1_scores)
            mean_bleu[distance_index, gamma_index] = np.mean(bleu_scores)

            std_sbert_semantic_sim[distance_index, gamma_index] = np.std(
                sbert_semantic_sim_scores, ddof=1
            ) / np.sqrt(n_test_samples)
            std_bleu_1[distance_index, gamma_index] = np.std(
                bleu1_scores, ddof=1
            ) / np.sqrt(n_test_samples)
            # std_semantic_sim[distance_index, gamma_index] = np.std(
            #     gpt_scores, ddof=1
            # ) / np.sqrt(n_test_samples)
            std_bleu[distance_index, gamma_index] = np.std(
                bleu_scores, ddof=1
            ) / np.sqrt(n_test_samples)

            np.save(
                os.path.join(results_dir, f"conventional_mean_semantic_sim_{args.channel_type}.npy"),
                mean_semantic_sim,
            )
            np.save(
                os.path.join(
                    results_dir, f"conventional_mean_sbert_semantic_sim_{args.channel_type}.npy"
                ),
                mean_sbert_semantic_sim,
            )
            np.save(
                os.path.join(results_dir, f"conventional_mean_bleu_1_{args.channel_type}.npy"),
                mean_bleu_1,
            )
            np.save(
                os.path.join(results_dir, f"conventional_mean_bleu_{args.channel_type}.npy"), mean_bleu
            )

            np.save(
                os.path.join(results_dir, f"conventional_std_semantic_sim_{args.channel_type}.npy"),
                std_semantic_sim,
            )
            np.save(
                os.path.join(results_dir, f"conventional_std_sbert_semantic_sim_{args.channel_type}.npy"),
                std_sbert_semantic_sim,
            )
            np.save(
                os.path.join(results_dir, f"conventional_std_bleu_1_{args.channel_type}.npy"), std_bleu_1
            )
            np.save(os.path.join(results_dir, f"conventional_std_bleu_{args.channel_type}.npy"), std_bleu)

            # df = pd.DataFrame(
            #     records,
            #     columns=[
            #         "d_sd",
            #         "Gamma",
            #         "Sentence 1",
            #         "Sentence 2",
            #         "Semantic Similarity Score",
            #         "BLEU 1 Gram Score",
            #         "BLEU Score",
            #         "SBERT Semantic Score",
            #     ],
            # )
            # df.to_excel(os.path.join(results_dir, f"conventional_output_{args.channel_type}.xlsx"), index=False)
