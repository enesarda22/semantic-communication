import argparse
import numpy as np
import matplotlib.pyplot as plt

from nltk.translate.bleu_score import sentence_bleu

import torch
from torch.nn import functional as F

from semantic_communication.models.transceiver import (
    TxRelayChannelModel,
    TxRelayRxChannelModel,
    Transceiver,
    ChannelEncoder,
    RelayChannelBlock,
)
from semantic_communication.utils.general import (
    get_device,
    set_seed,
    add_semantic_decoder_args,
    add_channel_model_args,
    add_data_args,
)
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.models.semantic_decoder import SemanticDecoder
from semantic_communication.utils.channel import init_channel


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
    parser.add_argument("--transceiver-path", type=str)
    add_semantic_decoder_args(parser)
    add_channel_model_args(parser)
    add_data_args(parser)

    # test args
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--gamma-list", nargs="+", type=float)
    parser.add_argument("--d-list", nargs="+", type=float)
    parser.add_argument("--n-test", default=20000, type=int)
    parser.add_argument("--semantic-similarity-threshold", default=0.8, type=float)
    parser.add_argument("--bleu-1-threshold", default=0.5, type=float)
    parser.add_argument("--bleu-2-threshold", default=0.5, type=float)
    parser.add_argument("--bleu-3-threshold", default=0.5, type=float)
    parser.add_argument("--bleu-4-threshold", default=0.5, type=float)
    args = parser.parse_args()

    device = get_device()
    set_seed()

    semantic_encoder = SemanticEncoder(max_length=args.max_length)
    data_handler = DataHandler(
        semantic_encoder=semantic_encoder,
        data_fp=args.data_fp,
        batch_size=args.batch_size,
    )

    # initialize models
    relay_decoder = SemanticDecoder(
        vocab_size=data_handler.vocab_size,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        n_embeddings=args.n_embeddings,
        block_size=args.max_length,
    ).to(device)

    tx_channel_enc = ChannelEncoder(
        nin=args.channel_block_input_dim,
        nout=args.channel_block_latent_dim,
    ).to(device)

    channel = init_channel(args.channel_type, args.sig_pow, args.alpha, args.noise_pow)
    tx_relay_channel_model = TxRelayChannelModel(
        nin=args.channel_block_input_dim,
        n_latent=args.channel_block_latent_dim,
        channel=channel,
    ).to(device)

    relay_channel_block = RelayChannelBlock(
        semantic_decoder=relay_decoder,
        tx_channel_enc=tx_channel_enc,
        tx_relay_channel_enc_dec=tx_relay_channel_model,
    ).to(device)

    receiver_decoder = SemanticDecoder(
        vocab_size=data_handler.vocab_size,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        n_embeddings=args.n_embeddings * 2,
        block_size=args.max_length,
    ).to(device)

    tx_relay_rx_channel_model = TxRelayRxChannelModel(
        nin=args.channel_block_input_dim,
        n_latent=args.channel_block_latent_dim,
        channel=channel,
    ).to(device)

    transceiver = Transceiver(
        semantic_encoder=semantic_encoder,
        relay_channel_block=relay_channel_block,
        rx_semantic_decoder=receiver_decoder,
        tx_relay_rx_channel_enc_dec=tx_relay_rx_channel_model,
        encoder=data_handler.encoder,
    )
    transceiver_checkpoint = torch.load(args.transceiver_path, map_location=device)
    transceiver.load_state_dict(transceiver_checkpoint["model_state_dict"])

    mean_semantic_sim = np.zeros((len(args.d_list), len(args.gamma_list)))
    mean_bleu_1 = np.zeros((len(args.d_list), len(args.gamma_list)))
    mean_bleu_2 = np.zeros((len(args.d_list), len(args.gamma_list)))
    mean_bleu_3 = np.zeros((len(args.d_list), len(args.gamma_list)))
    mean_bleu_4 = np.zeros((len(args.d_list), len(args.gamma_list)))

    std_semantic_sim = np.zeros((len(args.d_list), len(args.gamma_list)))
    std_bleu_1 = np.zeros((len(args.d_list), len(args.gamma_list)))
    std_bleu_2 = np.zeros((len(args.d_list), len(args.gamma_list)))
    std_bleu_3 = np.zeros((len(args.d_list), len(args.gamma_list)))
    std_bleu_4 = np.zeros((len(args.d_list), len(args.gamma_list)))

    semantic_sim_efficiency = np.zeros((len(args.d_list), len(args.gamma_list)))
    bleu_1_efficiency = np.zeros((len(args.d_list), len(args.gamma_list)))
    bleu_2_efficiency = np.zeros((len(args.d_list), len(args.gamma_list)))
    bleu_3_efficiency = np.zeros((len(args.d_list), len(args.gamma_list)))
    bleu_4_efficiency = np.zeros((len(args.d_list), len(args.gamma_list)))

    # For each d_sd
    for distance_index, d_sd in enumerate(args.d_list):

        # For each gamma in gamma list
        for gamma_index, gamma in enumerate(args.gamma_list):
            print(f"Simulating for distance: {d_sd}  - Gamma: {gamma}")

            cosine_scores = []
            bleu1_scores = []
            bleu2_scores = []
            bleu3_scores = []
            bleu4_scores = []

            d_sr = d_sd * gamma
            d_rd = d_sd - d_sr

            transceiver.eval()

            time_slot = 0
            semantic_similarity_num_correct_sentences = 0
            bleu_1_num_correct_sentences = 0
            bleu_2_num_correct_sentences = 0
            bleu_3_num_correct_sentences = 0
            bleu_4_num_correct_sentences = 0


            for b in data_handler.test_dataloader:
                xb = b[0].to(device)
                targets = data_handler.encode_token_ids(xb)
                attention_mask = b[1].to(device)
                time_slot += torch.sum(attention_mask).item()

                B, T = xb.shape
                with torch.no_grad():
                    logits, _ = transceiver(
                        xb, attention_mask, targets[:, 1:], d_sd, d_sr, d_rd
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
                        ids=targets,
                        attention_mask=attention_mask,
                        skip_special_tokens=True,
                    )

                    for s1, s2 in zip(original_sentences, predicted_sentences):
                        cosine_score = semantic_similarity_score([s1], [s2])[0][0].item()
                        bleu1_score = bleu_1gram(s1, s2)
                        bleu2_score = bleu_2gram(s1, s2)
                        bleu3_score = bleu_3gram(s1, s2)
                        bleu4_score = bleu_4gram(s1, s2)

                        if args.semantic_similarity_threshold <= cosine_score:
                            semantic_similarity_num_correct_sentences += 1

                        if args.bleu_1_threshold <= bleu1_score:
                            bleu_1_num_correct_sentences += 1

                        if args.bleu_2_threshold <= bleu2_score:
                            bleu_2_num_correct_sentences += 1

                        if args.bleu_3_threshold <= bleu3_score:
                            bleu_3_num_correct_sentences += 1

                        if args.bleu_4_threshold <= bleu4_score:
                            bleu_4_num_correct_sentences += 1

                        cosine_scores.append(cosine_score)
                        bleu1_scores.append(bleu1_score)
                        bleu2_scores.append(bleu2_score)
                        bleu3_scores.append(bleu3_score)
                        bleu4_scores.append(bleu4_score)
                if len(cosine_scores) > args.n_test:
                    break

            time_slot = time_slot * 2

            semantic_sim_efficiency[distance_index, gamma_index] = semantic_similarity_num_correct_sentences / time_slot
            bleu_1_efficiency[distance_index, gamma_index] = bleu_1_num_correct_sentences / time_slot
            bleu_2_efficiency[distance_index, gamma_index] = bleu_2_num_correct_sentences / time_slot
            bleu_3_efficiency[distance_index, gamma_index] = bleu_3_num_correct_sentences / time_slot
            bleu_4_efficiency[distance_index, gamma_index] = bleu_4_num_correct_sentences / time_slot

            mean_semantic_sim[distance_index, gamma_index] = np.mean(cosine_scores)
            mean_bleu_1[distance_index, gamma_index] = np.mean(bleu1_scores)
            mean_bleu_2[distance_index, gamma_index] = np.mean(bleu2_scores)
            mean_bleu_3[distance_index, gamma_index] = np.mean(bleu3_scores)
            mean_bleu_4[distance_index, gamma_index] = np.mean(bleu4_scores)

            std_semantic_sim[distance_index, gamma_index] = np.std(cosine_scores, ddof=1) / np.sqrt(len(cosine_scores))
            std_bleu_1[distance_index, gamma_index] = np.std(bleu1_scores, ddof=1) / np.sqrt(len(bleu1_scores))
            std_bleu_2[distance_index, gamma_index] = np.std(bleu2_scores, ddof=1) / np.sqrt(len(bleu2_scores))
            std_bleu_3[distance_index, gamma_index] = np.std(bleu3_scores, ddof=1) / np.sqrt(len(bleu3_scores))
            std_bleu_4[distance_index, gamma_index] = np.std(bleu4_scores, ddof=1) / np.sqrt(len(bleu4_scores))

    np.save("spf_mean_semantic_sim.npy", mean_semantic_sim)
    np.save("spf_mean_bleu_1.npy", mean_bleu_1)
    np.save("spf_mean_bleu_2.npy", mean_bleu_2)
    np.save("spf_mean_bleu_3.npy", mean_bleu_3)
    np.save("spf_mean_bleu_4.npy", mean_bleu_4)

    np.save("spf_std_semantic_sim.npy", std_semantic_sim)
    np.save("spf_std_bleu_1.npy", std_bleu_1)
    np.save("spf_std_bleu_2.npy", std_bleu_2)
    np.save("spf_std_bleu_3.npy", std_bleu_3)
    np.save("spf_std_bleu_4.npy", std_bleu_4)

    np.save("spf_efficiency_semantic_sim.npy", semantic_sim_efficiency)
    np.save("spf_efficiency_bleu_1.npy", bleu_1_efficiency)
    np.save("spf_efficiency_bleu_2.npy", bleu_2_efficiency)
    np.save("spf_efficiency_bleu_3.npy", bleu_3_efficiency)
    np.save("spf_efficiency_bleu_4.npy", bleu_4_efficiency)

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