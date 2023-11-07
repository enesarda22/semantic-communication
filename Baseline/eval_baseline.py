import numpy as np
import matplotlib.pyplot as plt

from nltk.translate.bleu_score import sentence_bleu
from utils.general import get_device, set_seed
from baseline_models.tx_relay_rx_models import Tx_Relay, Tx_Relay_Rx

from data_processing.data_handler import DataHandler
from data_processing.semantic_encoder import SemanticEncoder
from utils.channel import AWGN, Rayleigh
import torch
import argparse
from torch.nn import functional as F



def semantic_similarity_score(target_sentences, received_sentences):
    target_emb = semantic_encoder(messages=target_sentences)
    received_emb = semantic_encoder(messages=received_sentences)
    scores = F.cosine_similarity(target_emb, received_emb)

    return scores


def bleu_1gram(target_sentences, received_sentences):
    # score = []
    # for (sent1, sent2) in zip(target_sentences, received_sentences):
    #     sent1 = sent1.split()
    #     sent2 = sent2.split()
    #     score.append(sentence_bleu([sent1], sent2,
    #                                weights=(1, 0, 0, 0)))
    return sentence_bleu(
        [target_sentences], received_sentences, weights=(1, 0, 0, 0)
    )


def bleu_2gram(target_sentences, received_sentences):
    # score = []
    # for (sent1, sent2) in zip(target_sentences, received_sentences):
    #     sent1 = sent1.split()
    #     sent2 = sent2.split()
    #     score.append(sentence_bleu([sent1], sent2,
    #                                weights=(0, 1, 0, 0)))
    return sentence_bleu(
        [target_sentences], received_sentences, weights=(0, 1, 0, 0)
    )


def bleu_3gram(target_sentences, received_sentences):
    # score = []
    # for (sent1, sent2) in zip(target_sentences, received_sentences):
    #     sent1 = sent1.split()
    #     sent2 = sent2.split()
    #     score.append(sentence_bleu([sent1], sent2,
    #                                weights=(0, 0, 1, 0)))
    return sentence_bleu(
        [target_sentences], received_sentences, weights=(0, 0, 1, 0)
    )


def bleu_4gram(target_sentences, received_sentences):
    # score = []
    # for (sent1, sent2) in zip(target_sentences, received_sentences):
    #     sent1 = sent1.split()
    #     sent2 = sent2.split()
    #     score.append(sentence_bleu([sent1], sent2,
    #                                weights=(0, 0, 0, 1)))
    return sentence_bleu(
        [target_sentences], received_sentences, weights=(0, 0, 0, 1)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tx-relay-path", type=str)
    parser.add_argument("--tx-relay-rx-path", type=str)

    parser.add_argument("--SNR-list", nargs="+", type=int)

    parser.add_argument("--checkpoint-path", default="checkpoints", type=str)
    parser.add_argument("--n-samples", default=10000, type=int)
    parser.add_argument("--train-size", default=0.9, type=float)
    parser.add_argument("--max-length", default=30, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--n-epochs", default=10, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--n-blocks", default=1, type=int)
    parser.add_argument("--n-heads", default=4, type=int)
    parser.add_argument("--n-embeddings", default=384, type=int)

    # New args
    parser.add_argument("--channel-block-input-dim", default=384, type=int)
    parser.add_argument("--channel-block-latent-dim", default=128, type=int)
    parser.add_argument("--val-size", default=0.2, type=float)
    parser.add_argument("--sig-pow", default=1.0, type=float)
    parser.add_argument("--SNR-diff", default=3, type=int)
    parser.add_argument("--channel-type", default="AWGN", type=str)
    args = parser.parse_args()

    device = get_device()
    set_seed()
    # Create Data handler
    semantic_encoder = SemanticEncoder(max_length=args.max_length)
    data_handler = DataHandler(
        semantic_encoder=semantic_encoder,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        train_size=args.train_size,
        val_size=args.val_size,
    )
    data_handler.load_data()

    # Create Channels
    if args.channel_type == "AWGN":
        tx_rx_channel = AWGN(
            int(args.SNR_list[0]) - args.SNR_diff, args.sig_pow
        )
        tx_relay_channel = AWGN(int(args.SNR_list[0]), args.sig_pow)
        relay_rx_channel = AWGN(int(args.SNR_list[0]), args.sig_pow)

    else:
        tx_rx_channel = Rayleigh(
            int(args.SNR_list[0]) - args.SNR_diff, args.sig_pow
        )
        tx_relay_channel = Rayleigh(int(args.SNR_list[0]), args.sig_pow)
        relay_rx_channel = Rayleigh(int(args.SNR_list[0]), args.sig_pow)

    num_classes = data_handler.vocab_size

    # Create Transceiver
    tx_relay_model = Tx_Relay(num_classes, n_emb=args.channel_block_input_dim, n_latent=args.channel_block_latent_dim, channel=tx_relay_channel).to(device)
    tx_relay_checkpoint = torch.load(args.tx_relay_path)
    tx_relay_model.load_state_dict(tx_relay_checkpoint["model_state_dict"])

    tx_relay_rx_model = Tx_Relay_Rx(num_classes, args.channel_block_input_dim, args.channel_block_latent_dim, tx_rx_channel, relay_rx_channel,tx_relay_model).to(device)
    tx_relay_rx_checkpoint = torch.load(args.tx_relay_rx_path)
    tx_relay_rx_model.load_state_dict(tx_relay_rx_checkpoint["model_state_dict"])

    semantic_sim = []
    bleu_1 = []
    bleu_2 = []
    bleu_3 = []
    bleu_4 = []
    for SNR in args.SNR_list:
        print("Simulating for SNR: " + str(SNR))
        # Create Channels
        if args.channel_type == "AWGN":
            tx_rx_channel = AWGN(int(SNR) - args.SNR_diff, args.sig_pow)
            tx_relay_channel = AWGN(int(SNR), args.sig_pow)
            relay_rx_channel = AWGN(int(SNR), args.sig_pow)

        else:
            tx_rx_channel = Rayleigh(int(SNR) - args.SNR_diff, args.sig_pow)
            tx_relay_channel = Rayleigh(int(SNR), args.sig_pow)
            relay_rx_channel = Rayleigh(int(SNR), args.sig_pow)

        tx_relay_rx_model.tx_rx_channel = tx_rx_channel
        tx_relay_rx_model.relay_rx_channel = relay_rx_channel
        tx_relay_rx_model.tx_relay_model.channel = tx_relay_channel

        cosine_scores = []
        bleu1_scores = []
        bleu2_scores = []
        bleu3_scores = []
        bleu4_scores = []

        tx_relay_rx_model.eval()
        for b in data_handler.test_dataloader:
            xb = b[0].to(device)
            attention_mask = b[1].to(device)

            B, T = xb.shape

            with torch.no_grad():
                logits, _ = tx_relay_rx_model(xb, attention_mask)
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
                    cosine_scores.append(
                        semantic_similarity_score([s1], [s2])[0][0]
                    )

                    bleu1_scores.append(bleu_1gram(s1, s2))
                    bleu2_scores.append(bleu_2gram(s1, s2))
                    bleu3_scores.append(bleu_3gram(s1, s2))
                    bleu4_scores.append(bleu_4gram(s1, s2))

        semantic_sim.append(np.mean(cosine_scores))
        bleu_1.append(np.mean(bleu1_scores))
        bleu_2.append(np.mean(bleu2_scores))
        bleu_3.append(np.mean(bleu3_scores))
        bleu_4.append(np.mean(bleu4_scores))

    snr_np = np.array(args.SNR_list).astype(int)

    plt.figure()
    plt.plot(args.SNR_list, semantic_sim)
    plt.grid()
    plt.xlabel("Channel SNR (dB)")
    plt.ylabel("Semantic Similarity")
    plt.xticks(np.arange(np.min(snr_np), np.max(snr_np), 3))
    plt.title("Semantic Similarity v. Channel SNR (dB)")
    plt.savefig("SemanticSimilarty_v_SNR.png", dpi=400)

    plt.figure()
    plt.plot(args.SNR_list, bleu_1)
    plt.grid()
    plt.xlabel("Channel SNR (dB)")
    plt.ylabel("BLEU 1-gram")
    plt.xticks(np.arange(np.min(snr_np), np.max(snr_np), 3))
    plt.title("BLEU 1-gram v. Channel SNR (dB)")
    plt.savefig("BLEU1gram_v_SNR.png", dpi=400)

    plt.figure()
    plt.plot(args.SNR_list, bleu_2)
    plt.grid()
    plt.xlabel("Channel SNR (dB)")
    plt.ylabel("BLEU 2-gram")
    plt.xticks(np.arange(np.min(snr_np), np.max(snr_np), 3))
    plt.title("BLEU 2-gram v. Channel SNR (dB)")
    plt.savefig("BLEU2gam_v_SNR.png", dpi=400)

    plt.figure()
    plt.plot(args.SNR_list, bleu_3)
    plt.grid()
    plt.xlabel("Channel SNR (dB)")
    plt.ylabel("BLEU 3-gram")
    plt.xticks(np.arange(np.min(snr_np), np.max(snr_np), 3))
    plt.title("BLEU 3-gram v. Channel SNR (dB)")
    plt.savefig("BLEU3gram_v_SNR.png", dpi=400)

    plt.figure()
    plt.plot(args.SNR_list, bleu_4)
    plt.grid()
    plt.xlabel("Channel SNR (dB)")
    plt.ylabel("BLEU 4-gram")
    plt.xticks(np.arange(np.min(snr_np), np.max(snr_np), 3))
    plt.title("BLEU 4-gram v. Channel SNR (dB)")
    plt.savefig("BLEU4gram_v_SNR.png", dpi=400)

    with open('semantic_sim.npy', 'wb') as f:
        np.save(f, semantic_sim)

    with open('bleu_1.npy', 'wb') as f:
        np.save(f, bleu_1)

    with open('bleu_2.npy', 'wb') as f:
        np.save(f, bleu_2)

    with open('bleu_3.npy', 'wb') as f:
        np.save(f, bleu_3)

    with open('bleu_4.npy', 'wb') as f:
        np.save(f, bleu_4)
