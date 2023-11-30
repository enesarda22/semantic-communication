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
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--gamma-list", nargs="+", type=float)
    parser.add_argument("--d", type=float)

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

    semantic_sim = []
    bleu_1 = []
    bleu_2 = []
    bleu_3 = []
    bleu_4 = []

    d_sd = args.d

    for gamma in args.gamma_list:
        print("Simulating for distance: " + str(gamma * args.d))

        cosine_scores = []
        bleu1_scores = []
        bleu2_scores = []
        bleu3_scores = []
        bleu4_scores = []

        d_sr = d_sd * gamma
        d_rd = d_sd - d_sr

        tx_relay_rx_model.eval()
        for b in data_handler.test_dataloader:
            xb = b[0].to(device)
            attention_mask = b[1].to(device)
            xb = data_handler.encode_token_ids(xb)

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
                    cosine_scores.append(semantic_similarity_score([s1], [s2])[0][0])

                    bleu1_scores.append(bleu_1gram(s1, s2))
                    bleu2_scores.append(bleu_2gram(s1, s2))
                    bleu3_scores.append(bleu_3gram(s1, s2))
                    bleu4_scores.append(bleu_4gram(s1, s2))
            if len(cosine_scores) > 5000:
                break

        semantic_sim.append(np.mean([i.tolist() for i in cosine_scores]))
        bleu_1.append(np.mean(bleu1_scores))
        bleu_2.append(np.mean(bleu2_scores))
        bleu_3.append(np.mean(bleu3_scores))
        bleu_4.append(np.mean(bleu4_scores))

    d_sr_np = np.array(args.gamma_list) * args.d
    ticks = 0.2

    plt.figure()
    plt.plot(d_sr_np, semantic_sim)
    plt.grid()
    plt.xlabel("S-R Distance")
    plt.ylabel("Semantic Similarity")
    # plt.xticks(np.arange(np.min(distance_np), np.max(distance_np), ticks))
    plt.title("Semantic Similarity v. S-R Distance Ratio")
    plt.savefig("SemanticSimilarty_v_distance.png", dpi=400)

    plt.figure()
    plt.plot(d_sr_np, bleu_1)
    plt.grid()
    plt.xlabel("S-R Distance")
    plt.ylabel("BLEU 1-gram")
    # plt.xticks(np.arange(np.min(distance_np), np.max(distance_np), ticks))
    plt.title("BLEU 1-gram v. S-R Distance")
    plt.savefig("BLEU1gram_v_distance.png", dpi=400)

    plt.figure()
    plt.plot(d_sr_np, bleu_2)
    plt.grid()
    plt.xlabel("S-R Distance")
    plt.ylabel("BLEU 2-gram")
    # plt.xticks(np.arange(np.min(distance_np), np.max(distance_np), ticks))
    plt.title("BLEU 2-gram v. S-R Distance")
    plt.savefig("BLEU2gam_v_distance.png", dpi=400)

    plt.figure()
    plt.plot(d_sr_np, bleu_3)
    plt.grid()
    plt.xlabel("S-R Distance")
    plt.ylabel("BLEU 3-gram")
    # plt.xticks(np.arange(np.min(distance_np), np.max(distance_np), ticks))
    plt.title("BLEU 3-gram v. S-R Distance")
    plt.savefig("BLEU3gram_v_distance.png", dpi=400)

    plt.figure()
    plt.plot(d_sr_np, bleu_4)
    plt.grid()
    plt.xlabel("S-R Distance")
    plt.ylabel("BLEU 4-gram")
    # plt.xticks(np.arange(np.min(distance_np), np.max(distance_np), ticks))
    plt.title("BLEU 4-gram v. S-R Distance")
    plt.savefig("BLEU4gram_v_distance.png", dpi=400)

    np.save("conventional_semantic_sim.npy", semantic_sim)

    np.save("conventional_bleu_1.npy", bleu_1)
    np.save("conventional_bleu_2.npy", bleu_2)
    np.save("conventional_bleu_3.npy", bleu_3)
    np.save("conventional_bleu_4.npy", bleu_4)
