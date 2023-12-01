import numpy as np
import matplotlib.pyplot as plt

from nltk.translate.bleu_score import sentence_bleu
from semantic_communication.utils.general import (
    get_device,
    set_seed,
    add_semantic_decoder_args,
    add_channel_model_args,
    add_data_args,
    load_model,
)
from semantic_communication.models.baseline_models import Tx_Relay, Tx_Relay_Rx
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.utils.channel import init_channel
import torch
import argparse
from torch.nn import functional as F


def semantic_similarity_score(target_sentences, received_sentences):
    target_emb = semantic_encoder(messages=target_sentences)
    received_emb = semantic_encoder(messages=received_sentences)
    scores = F.cosine_similarity(target_emb, received_emb)
    return scores


def bleu_1gram(target_sentences, received_sentences):
    return sentence_bleu([target_sentences], received_sentences, weights=(1, 0, 0, 0))


def bleu_2gram(target_sentences, received_sentences):
    return sentence_bleu([target_sentences], received_sentences, weights=(0, 1, 0, 0))


def bleu_3gram(target_sentences, received_sentences):
    return sentence_bleu([target_sentences], received_sentences, weights=(0, 0, 1, 0))


def bleu_4gram(target_sentences, received_sentences):
    return sentence_bleu([target_sentences], received_sentences, weights=(0, 0, 0, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model args
    parser.add_argument("--tx-relay-path", type=str)
    parser.add_argument("--tx-relay-rx-path", type=str)

    add_semantic_decoder_args(parser)
    add_channel_model_args(parser)
    add_data_args(parser)

    # test args
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--gamma-list", nargs="+", type=float)
    parser.add_argument("--d-list", nargs="+", type=float)
    parser.add_argument("--n-test", default=10000, type=int)
    parser.add_argument("--semantic-similarity-threshold", default=0.8, type=float)
    parser.add_argument("--bleu-1-threshold", default=0.5, type=float)
    parser.add_argument("--bleu-3-threshold", default=0.5, type=float)

    args = parser.parse_args()
    device = get_device()
    set_seed()

    semantic_encoder = SemanticEncoder(max_length=args.max_length)
    data_handler = DataHandler(
        semantic_encoder=semantic_encoder,
        batch_size=args.batch_size,
        data_fp=args.data_fp,
    )

    channel = init_channel(args.channel_type, args.sig_pow, args.alpha, args.noise_pow)
    num_classes = data_handler.vocab_size

    # Create Transceiver
    tx_relay_model = Tx_Relay(
        num_classes,
        n_emb=args.channel_block_input_dim,
        n_latent=args.channel_block_latent_dim,
        channel=channel,
        entire_network_train=1,
    ).to(device)
    load_model(tx_relay_model, args.tx_relay_path)

    tx_relay_rx_model = Tx_Relay_Rx(
        num_classes,
        args.channel_block_input_dim,
        args.channel_block_latent_dim,
        channel,
        tx_relay_model,
    ).to(device)
    load_model(tx_relay_rx_model, args.tx_relay_rx_path)

    mean_semantic_sim = np.zeros((len(args.d_list), len(args.gamma_list)))
    mean_bleu_1 = np.zeros((len(args.d_list), len(args.gamma_list)))
    mean_bleu_3 = np.zeros((len(args.d_list), len(args.gamma_list)))

    std_semantic_sim = np.zeros((len(args.d_list), len(args.gamma_list)))
    std_bleu_1 = np.zeros((len(args.d_list), len(args.gamma_list)))
    std_bleu_3 = np.zeros((len(args.d_list), len(args.gamma_list)))

    semantic_sim_efficiency = np.zeros((len(args.d_list), len(args.gamma_list)))
    bleu_1_efficiency = np.zeros((len(args.d_list), len(args.gamma_list)))
    bleu_3_efficiency = np.zeros((len(args.d_list), len(args.gamma_list)))

    semantic_sim_efficiency_se = np.zeros((len(args.d_list), len(args.gamma_list)))
    bleu_1_efficiency_se = np.zeros((len(args.d_list), len(args.gamma_list)))
    bleu_3_efficiency_se = np.zeros((len(args.d_list), len(args.gamma_list)))

    # For each d_sd
    for distance_index, d_sd in enumerate(args.d_list):

        # For each gamma in gamma list
        for gamma_index, gamma in enumerate(args.gamma_list):
            print(f"Simulating for distance: {d_sd}  - Gamma: {gamma}")

            cosine_scores = []
            bleu1_scores = []
            bleu3_scores = []

            d_sr = d_sd * gamma
            d_rd = d_sd - d_sr

            tx_relay_rx_model.eval()

            time_slot = 0
            semantic_similarity_num_correct_sentences = 0
            bleu_1_num_correct_sentences = 0
            bleu_3_num_correct_sentences = 0

            for b in data_handler.test_dataloader:
                xb = b[0].to(device)
                attention_mask = b[1].to(device)
                xb = data_handler.encode_token_ids(xb)
                time_slot += (torch.sum(attention_mask) - attention_mask.shape[0]).item()

                B, T = xb.shape
                with torch.no_grad():
                    logits, _ = tx_relay_rx_model(
                        xb[:, 1:], attention_mask[:, 1:], d_sd, d_sr, d_rd
                    )
                    probs = F.softmax(logits, dim=-1)
                    predicted_ids = (torch.argmax(probs, dim=-1)).reshape(
                        B, args.max_length
                    )

                    end_token_id = data_handler.encoder.transform([102])[0]
                    end_prediction_idx = torch.argmax(
                        predicted_ids.eq(end_token_id).double(), dim=1
                    )

                    # zero means no end token prediction
                    end_prediction_idx[end_prediction_idx == 0] = T - 1

                    # prediction mask is created based on end token predictions
                    pred_mask = (torch.arange(T - 1).to(device)).le(
                        end_prediction_idx.view(-1, 1)
                    )

                    predicted_sentences = data_handler.get_tokens(
                        ids=predicted_ids,
                        attention_mask=pred_mask,
                        skip_special_tokens=True,
                    )

                    original_sentences = data_handler.get_tokens(
                        ids=xb,
                        attention_mask=attention_mask,
                        skip_special_tokens=True,
                    )

                    for s1, s2 in zip(original_sentences, predicted_sentences):
                        cosine_score = semantic_similarity_score([s1], [s2])[0][0]. item()
                        bleu1_score = bleu_1gram(s1, s2)
                        bleu3_score = bleu_3gram(s1, s2)

                        if args.semantic_similarity_threshold <= cosine_score:
                            semantic_similarity_num_correct_sentences += 1

                        if args.bleu_1_threshold <= bleu1_score:
                            bleu_1_num_correct_sentences += 1

                        if args.bleu_3_threshold <= bleu3_score:
                            bleu_3_num_correct_sentences += 1

                        cosine_scores.append(cosine_score)
                        bleu1_scores.append(bleu1_score)
                        bleu3_scores.append(bleu3_score)
                if len(cosine_scores) > args.n_test:
                    break

            time_slot = time_slot * 2

            semantic_sim_efficiency[distance_index, gamma_index] = semantic_similarity_num_correct_sentences / time_slot
            bleu_1_efficiency[distance_index, gamma_index] = bleu_1_num_correct_sentences / time_slot
            bleu_3_efficiency[distance_index, gamma_index] = bleu_3_num_correct_sentences / time_slot

            semantic_sim_efficiency_se[distance_index, gamma_index] = ((args.n_test ** 0.5) / time_slot) * (
                        1 - semantic_similarity_num_correct_sentences / args.n_test)
            bleu_1_efficiency_se[distance_index, gamma_index] = ((args.n_test ** 0.5) / time_slot) * (
                        1 - bleu_1_num_correct_sentences / args.n_test)
            bleu_3_efficiency_se[distance_index, gamma_index] = ((args.n_test ** 0.5) / time_slot) * (
                        1 - bleu_3_num_correct_sentences / args.n_test)

            mean_semantic_sim[distance_index, gamma_index] = np.mean(cosine_scores)
            mean_bleu_1[distance_index, gamma_index] = np.mean(bleu1_scores)
            mean_bleu_3[distance_index, gamma_index] = np.mean(bleu3_scores)

            std_semantic_sim[distance_index, gamma_index] = np.std(cosine_scores, ddof=1) / np.sqrt(len(cosine_scores))
            std_bleu_1[distance_index, gamma_index] = np.std(bleu1_scores, ddof=1) / np.sqrt(len(bleu1_scores))
            std_bleu_3[distance_index, gamma_index] = np.std(bleu3_scores, ddof=1) / np.sqrt(len(bleu3_scores))

    np.save("conventional_mean_semantic_sim.npy", mean_semantic_sim)
    np.save("conventional_mean_bleu_1.npy", mean_bleu_1)
    np.save("conventional_mean_bleu_3.npy", mean_bleu_3)

    np.save("conventional_std_semantic_sim.npy", std_semantic_sim)
    np.save("conventional_std_bleu_1.npy", std_bleu_1)
    np.save("conventional_std_bleu_3.npy", std_bleu_3)

    np.save("conventional_efficiency_semantic_sim.npy", semantic_sim_efficiency)
    np.save("conventional_efficiency_bleu_1.npy", bleu_1_efficiency)
    np.save("conventional_efficiency_bleu_3.npy", bleu_3_efficiency)

    np.save("conventional_efficiency_semantic_sim.npy", semantic_sim_efficiency_se)
    np.save("conventional_efficiency_bleu_1.npy", bleu_1_efficiency_se)
    np.save("conventional_efficiency_bleu_3.npy", bleu_3_efficiency_se)

    # d_sr_np = np.array(args.gamma_list) * args.d
    #
    # plt.figure()
    # plt.plot(d_sr_np, mean_semantic_sim)
    # plt.grid()
    # plt.xlabel("S-R Distance")
    # plt.ylabel("Semantic Similarity")
    # plt.title("Semantic Similarity v. S-R Distance Ratio")
    # plt.savefig("SemanticSimilarty_v_distance.png", dpi=400)
    #
    # plt.figure()
    # plt.plot(d_sr_np, mean_bleu_1)
    # plt.grid()
    # plt.xlabel("S-R Distance")
    # plt.ylabel("BLEU 1-gram")
    # plt.title("BLEU 1-gram v. S-R Distance")
    # plt.savefig("BLEU1gram_v_distance.png", dpi=400)
    #
    # plt.figure()
    # plt.plot(d_sr_np, mean_bleu_2)
    # plt.grid()
    # plt.xlabel("S-R Distance")
    # plt.ylabel("BLEU 2-gram")
    # plt.title("BLEU 2-gram v. S-R Distance")
    # plt.savefig("BLEU2gam_v_distance.png", dpi=400)
    #
    # plt.figure()
    # plt.plot(d_sr_np, mean_bleu_3)
    # plt.grid()
    # plt.xlabel("S-R Distance")
    # plt.ylabel("BLEU 3-gram")
    # plt.title("BLEU 3-gram v. S-R Distance")
    # plt.savefig("BLEU3gram_v_distance.png", dpi=400)
    #
    # plt.figure()
    # plt.plot(d_sr_np, mean_bleu_4)
    # plt.grid()
    # plt.xlabel("S-R Distance")
    # plt.ylabel("BLEU 4-gram")
    # plt.title("BLEU 4-gram v. S-R Distance")
    # plt.savefig("BLEU4gram_v_distance.png", dpi=400)
