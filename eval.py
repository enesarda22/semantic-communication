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
    return sentence_bleu(
        [target_sentences], received_sentences, weights=(1, 0, 0, 0)
    )


def bleu_2gram(target_sentences, received_sentences):
    return sentence_bleu(
        [target_sentences], received_sentences, weights=(0, 1, 0, 0)
    )


def bleu_3gram(target_sentences, received_sentences):
    return sentence_bleu(
        [target_sentences], received_sentences, weights=(0, 0, 1, 0)
    )


def bleu_4gram(target_sentences, received_sentences):
    return sentence_bleu(
        [target_sentences], received_sentences, weights=(0, 0, 0, 1)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model args
    parser.add_argument("--transceiver-path", type=str)
    add_semantic_decoder_args(parser)
    add_channel_model_args(parser)
    add_data_args(parser)

    # test args
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--distance-list", nargs="+", type=int)
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

    receiver_decoder = SemanticDecoder(
        vocab_size=data_handler.vocab_size,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        n_embeddings=args.n_embeddings * 2,
        block_size=args.max_length,
    ).to(device)

    channel = init_channel(args.channel_type, args.sig_pow)
    tx_relay_channel_model = TxRelayChannelModel(
        nin=args.channel_block_input_dim,
        n_latent=args.channel_block_latent_dim,
        channel=channel,
    ).to(device)
    tx_relay_rx_channel_model = TxRelayRxChannelModel(
        nin=args.channel_block_input_dim,
        n_latent=args.channel_block_latent_dim,
        channel=channel,
    ).to(device)

    transceiver = Transceiver(
        semantic_encoder,
        relay_decoder,
        receiver_decoder,
        tx_relay_channel_model,
        tx_relay_rx_channel_model,
        encoder=data_handler.encoder,
    )
    transceiver_checkpoint = torch.load(
        args.transceiver_path, map_location=device
    )
    transceiver.load_state_dict(transceiver_checkpoint["model_state_dict"])

    semantic_sim = []
    bleu_1 = []
    bleu_2 = []
    bleu_3 = []
    bleu_4 = []

    d_sd = args.d

    for distance_ratio in args.distance_list:
        print("Simulating for distance: " + str(distance_ratio * d_sd))

        cosine_scores = []
        bleu1_scores = []
        bleu2_scores = []
        bleu3_scores = []
        bleu4_scores = []
        d_sr = d_sd * distance_ratio
        d_rd = d_sd - d_sr

        transceiver.eval()
        for b in data_handler.test_dataloader:
            xb = b[0].to(device)
            targets = data_handler.encode_token_ids(xb)
            attention_mask = b[1].to(device)

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
                    cosine_scores.append(
                        semantic_similarity_score([s1], [s2])[0][0].tolist()
                    )

                    bleu1_scores.append(bleu_1gram(s1, s2))
                    bleu2_scores.append(bleu_2gram(s1, s2))
                    bleu3_scores.append(bleu_3gram(s1, s2))
                    bleu4_scores.append(bleu_4gram(s1, s2))
            if len(cosine_scores) > 5000:
                break

        semantic_sim.append(np.mean(cosine_scores))
        bleu_1.append(np.mean(bleu1_scores))
        bleu_2.append(np.mean(bleu2_scores))
        bleu_3.append(np.mean(bleu3_scores))
        bleu_4.append(np.mean(bleu4_scores))

    distance_np = np.array(args.distance_list)
    ticks = 0.2

    plt.figure()
    plt.plot(args.distance_list, semantic_sim)
    plt.grid()
    plt.xlabel("Distance Ratio")
    plt.ylabel("Semantic Similarity")
    plt.xticks(np.arange(np.min(distance_np), np.max(distance_np), ticks))
    plt.title("Semantic Similarity v. S-R Distance Ratio")
    plt.savefig("SemanticSimilarty_v_distance.png", dpi=400)

    plt.figure()
    plt.plot(args.distance_list, bleu_1)
    plt.grid()
    plt.xlabel("Distance Ratio")
    plt.ylabel("BLEU 1-gram")
    plt.xticks(np.arange(np.min(distance_np), np.max(distance_np), ticks))
    plt.title("BLEU 1-gram v. S-R Distance Ratio")
    plt.savefig("BLEU1gram_v_distance.png", dpi=400)

    plt.figure()
    plt.plot(args.distance_list, bleu_2)
    plt.grid()
    plt.xlabel("Distance Ratio")
    plt.ylabel("BLEU 2-gram")
    plt.xticks(np.arange(np.min(distance_np), np.max(distance_np), ticks))
    plt.title("BLEU 2-gram v. S-R Distance Ratio")
    plt.savefig("BLEU2gam_v_distance.png", dpi=400)

    plt.figure()
    plt.plot(args.distance_list, bleu_3)
    plt.grid()
    plt.xlabel("Distance Ratio")
    plt.ylabel("BLEU 3-gram")
    plt.xticks(np.arange(np.min(distance_np), np.max(distance_np), ticks))
    plt.title("BLEU 3-gram v. S-R Distance Ratio")
    plt.savefig("BLEU3gram_v_distance.png", dpi=400)

    plt.figure()
    plt.plot(args.SNR_list, bleu_4)
    plt.grid()
    plt.xlabel("Distance Ratio")
    plt.ylabel("BLEU 4-gram")
    plt.xticks(np.arange(np.min(distance_np), np.max(distance_np), ticks))
    plt.title("BLEU 4-gram v. S-R Distance Ratio")
    plt.savefig("BLEU4gram_v_distance.png", dpi=400)

    np.save("semantic_sim.npy", semantic_sim)

    np.save("bleu_1.npy", bleu_1)
    np.save("bleu_2.npy", bleu_2)
    np.save("bleu_3.npy", bleu_3)
    np.save("bleu_4.npy", bleu_4)