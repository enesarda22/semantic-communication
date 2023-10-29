import numpy as np
from sentence_transformers import SentenceTransformer, util
from semantic_communication.models.transceiver import (
    TxRelayChannelModel,
    TxRelayRxChannelModel,
    Transceiver,
)

import matplotlib.pyplot as plt

from nltk.translate.bleu_score import sentence_bleu
from semantic_communication.utils.general import get_device
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.models.semantic_decoder import SemanticDecoder
from semantic_communication.utils.channel import AWGN, Rayleigh
import torch
import argparse
from torch.nn import functional as F

def semantic_similarity_score(target_sentences, received_sentences, sbert):
    target_emb = sbert.encode(target_sentences)
    received_emb = sbert.encode(received_sentences)
    cosine_scores = util.cos_sim(target_emb, received_emb)

    return cosine_scores


def bleu_1gram(target_sentences, received_sentences):
    # score = []
    # for (sent1, sent2) in zip(target_sentences, received_sentences):
    #     sent1 = sent1.split()
    #     sent2 = sent2.split()
    #     score.append(sentence_bleu([sent1], sent2,
    #                                weights=(1, 0, 0, 0)))
    return sentence_bleu([target_sentences], received_sentences, weights=(1, 0, 0, 0))


def bleu_2gram(target_sentences, received_sentences):
    # score = []
    # for (sent1, sent2) in zip(target_sentences, received_sentences):
    #     sent1 = sent1.split()
    #     sent2 = sent2.split()
    #     score.append(sentence_bleu([sent1], sent2,
    #                                weights=(0, 1, 0, 0)))
    return sentence_bleu([target_sentences], received_sentences, weights=(0, 1, 0, 0))


def bleu_3gram(target_sentences, received_sentences):
    # score = []
    # for (sent1, sent2) in zip(target_sentences, received_sentences):
    #     sent1 = sent1.split()
    #     sent2 = sent2.split()
    #     score.append(sentence_bleu([sent1], sent2,
    #                                weights=(0, 0, 1, 0)))
    return sentence_bleu([target_sentences], received_sentences, weights=(0, 0, 1, 0))


def bleu_4gram(target_sentences, received_sentences):
    # score = []
    # for (sent1, sent2) in zip(target_sentences, received_sentences):
    #     sent1 = sent1.split()
    #     sent2 = sent2.split()
    #     score.append(sentence_bleu([sent1], sent2,
    #                                weights=(0, 0, 0, 1)))
    return sentence_bleu([target_sentences], received_sentences, weights=(0, 0, 0, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transceiver-path", type=str)
    parser.add_argument("--SNR-list", type=list)

    parser.add_argument("--checkpoint-path", default="checkpoints", type=str)
    parser.add_argument("--n-samples", default=10000, type=int)
    parser.add_argument("--train-size", default=0.9, type=float)
    parser.add_argument("--max-length", default=30, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--n-epochs", default=10, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--n-heads", default=4, type=int)
    parser.add_argument("--n-embeddings", default=384, type=int)

    # New args
    parser.add_argument("--val-size", default=0.2, type=float)
    parser.add_argument("--sig-pow", default=1.0, type=float)
    parser.add_argument("--SNR-min", default=3, type=int)
    parser.add_argument("--SNR-max", default=24, type=int)
    parser.add_argument("--SNR-step", default=3, type=int)
    parser.add_argument("--SNR-window", default=5, type=int)
    parser.add_argument("--SNR-diff", default=3, type=int)
    parser.add_argument("--channel-type", default="AWGN", type=str)
    args = parser.parse_args()

    device = get_device()

    # Create Data handler
    semantic_encoder = SemanticEncoder(max_length=args.max_length)
    data_handler = DataHandler(
        semantic_encoder=semantic_encoder,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        train_size=args.train_size,
        val_size=args.val_size
    )
    data_handler.load_data()

    # Create Channels
    if args.channel_type == "AWGN":
        tx_rx_channel = AWGN(int(args.SNR_list[0]) - args.SNR_diff, args.sig_pow)
        tx_relay_channel = AWGN(int(args.SNR_list[0]), args.sig_pow)
        relay_rx_channel = AWGN(int(args.SNR_list[0]), args.sig_pow)

    else:
        tx_rx_channel = Rayleigh(int(args.SNR_list[0]) - args.SNR_diff, args.sig_pow)
        tx_relay_channel = Rayleigh(int(args.SNR_list[0]), args.sig_pow)
        relay_rx_channel = Rayleigh(int(args.SNR_list[0]), args.sig_pow)

    # Create Transceiver
    relay_decoder = SemanticDecoder(
        vocab_size=data_handler.vocab_size,
        n_heads=args.n_heads,
        n_embeddings=args.n_embeddings,
        block_size=args.max_length,
    ).to(device)

    receiver_decoder = SemanticDecoder(
        vocab_size=data_handler.vocab_size,
        n_heads=args.n_heads,
        n_embeddings=args.n_embeddings,
        block_size=args.max_length,
    ).to(device)

    tx_relay_channel_model = TxRelayChannelModel(384, 128, tx_relay_channel).to(device)
    tx_relay_rx_channel_model = TxRelayRxChannelModel(
        384, 128, tx_rx_channel, relay_rx_channel
    ).to(device)

    transceiver = Transceiver(
        semantic_encoder,
        relay_decoder,
        receiver_decoder,
        tx_relay_channel_model,
        tx_relay_rx_channel_model,
    )
    transceiver_checkpoint = torch.load(args.transceiver_path)
    transceiver.load_state_dict(transceiver_checkpoint["model_state_dict"])

    semantic_sim = []
    bleu_1 = []
    bleu_2 = []
    bleu_3 = []
    bleu_4 = []
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
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

        transceiver.tx_relay_channel_enc_dec.channel=tx_relay_channel
        transceiver.tx_relay_rx_channel_enc_dec.channel_tx_rx=tx_rx_channel
        transceiver.tx_relay_rx_channel_enc_dec.channel_rel_rx = relay_rx_channel

        cosine_scores = []
        bleu1_scores = []
        bleu2_scores = []
        bleu3_scores =[]
        bleu4_scores = []

        transceiver.eval()
        for b in data_handler.test_dataloader:
            xb = b[0].to(device)
            attention_mask = b[1].to(device)

            with torch.no_grad():
                logits, loss = transceiver(xb, attention_mask)
                probs = F.softmax(logits, dim=-1)
                idx = (torch.argmax(probs, dim=-1)).reshape(xb.shape[0], args.max_length)

                for (sent1, sent2) in zip(xb, idx):
                    grnd = data_handler.get_text(sent1[1:].to("cpu"))
                    # TODO: THIS IS NOT WORKING PROPERLY
                    recv = data_handler.get_text(sent2.to("cpu"))

                    cosine_scores.append(semantic_similarity_score(grnd, recv, sbert)[0][0])

                    bleu1_scores.append(bleu_1gram(grnd, recv))
                    bleu2_scores.append(bleu_2gram(grnd, recv))
                    bleu3_scores.append(bleu_3gram(grnd, recv))
                    bleu4_scores.append(bleu_4gram(grnd, recv))

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
plt.xticks(np.arange(np.min(snr_np),np.max(snr_np),3))
plt.title("Semantic Similarity v. Channel SNR (dB)")
plt.savefig('SemanticSimilarty_v_SNR.png', dpi=400)

plt.figure()
plt.plot(args.SNR_list, bleu_1)
plt.grid()
plt.xlabel("Channel SNR (dB)")
plt.ylabel("BLEU 1-gram")
plt.xticks(np.arange(np.min(snr_np), np.max(snr_np), 3))
plt.title("BLEU 1-gram v. Channel SNR (dB)")
plt.savefig('BLEU1gram_v_SNR.png', dpi=400)

plt.figure()
plt.plot(args.SNR_list, bleu_2)
plt.grid()
plt.xlabel("Channel SNR (dB)")
plt.ylabel("BLEU 2-gram")
plt.xticks(np.arange(np.min(snr_np), np.max(snr_np), 3))
plt.title("BLEU 2-gram v. Channel SNR (dB)")
plt.savefig('BLEU2gam_v_SNR.png', dpi=400)

plt.figure()
plt.plot(args.SNR_list, bleu_3)
plt.grid()
plt.xlabel("Channel SNR (dB)")
plt.ylabel("BLEU 3-gram")
plt.xticks(np.arange(np.min(snr_np), np.max(snr_np), 3))
plt.title("BLEU 3-gram v. Channel SNR (dB)")
plt.savefig('BLEU3gram_v_SNR.png', dpi=400)

plt.figure()
plt.plot(args.SNR_list, bleu_4)
plt.grid()
plt.xlabel("Channel SNR (dB)")
plt.ylabel("BLEU 4-gram")
plt.xticks(np.arange(np.min(snr_np), np.max(snr_np), 3))
plt.title("BLEU 4-gram v. Channel SNR (dB)")
plt.savefig('BLEU4gram_v_SNR.png', dpi=400)

