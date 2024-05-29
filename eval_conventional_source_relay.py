import numpy as np
import matplotlib.pyplot as plt
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
    load_model,
)
import math
import itertools

from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.utils.channel import init_channel
from semantic_communication.models.semantic_encoder import SemanticEncoder
import re
import torch
import argparse
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer
from semantic_communication.utils.modulation import modulation
from scipy.special import erfc

class QPSK:
    def __init__(self, sig_pow=1.0):
        self.sig_pow = sig_pow

    def modulate(self, bit_sequence, h_re_hat=None, h_im_hat=None):
        if h_re_hat is None or h_im_hat is None:
            reshaped = np.reshape(bit_sequence, (int(len(bit_sequence) / 2), 2))
            const_val = np.sqrt(self.sig_pow) / np.sqrt(2)

            real = np.ones(len(reshaped)) * const_val
            real[reshaped[:, 1] == 1] = -const_val

            im = np.ones(len(reshaped)) * const_val
            im[reshaped[:, 0] == 1] = -const_val
            return real, im

        else:
            reshaped = np.reshape(bit_sequence, (int(len(bit_sequence) / 2), 2))
            re_1 = h_re_hat + h_im_hat
            re_2 = h_re_hat - h_im_hat

            im_1 = h_re_hat - h_im_hat
            im_2 = h_re_hat + h_im_hat

            c = np.sqrt(2 * (h_re_hat**2 + h_im_hat**2) / self.sig_pow)

            # 00
            re = np.ones(len(reshaped)) * re_1
            im = np.ones(len(reshaped)) * im_1

            # 10
            re[(np.sum(reshaped == ([1, 0]), axis=1)) == 2] = re_2[(np.sum(reshaped == ([1, 0]), axis=1)) == 2]
            im[(np.sum(reshaped == ([1, 0]), axis=1)) == 2] = -im_2[(np.sum(reshaped == ([1, 0]), axis=1)) == 2]

            # 01
            re[(np.sum(reshaped == ([0, 1]), axis=1)) == 2] = -re_2[(np.sum(reshaped == ([0, 1]), axis=1)) == 2]
            im[(np.sum(reshaped == ([0, 1]), axis=1)) == 2] = im_2[(np.sum(reshaped == ([0, 1]), axis=1)) == 2]

            # 11
            re[(np.sum(reshaped == ([1, 1]), axis=1)) == 2] = -re_1[(np.sum(reshaped == ([1, 1]), axis=1)) == 2]
            im[(np.sum(reshaped == ([1, 1]), axis=1)) == 2] = -im_1[(np.sum(reshaped == ([1, 1]), axis=1)) == 2]

            return re/c, im/c

    def demodulate(self, received_re, received_im):
        seconds = np.zeros(len(received_im))
        firsts = np.zeros(len(received_im))
        seconds[received_re <= 0] = 1
        firsts[received_im <= 0] = 1
        return np.array([firsts, seconds]).T.flatten()


class AWGN:
    def __init__(self, signal_power_constraint=1.0):
        self.signal_power_constraint = signal_power_constraint

    def __call__(self, x_re, x_im, SNR):
        linear_SNR = np.power(10, SNR / 10)
        noise_var = self.signal_power_constraint / linear_SNR
        noise_re = np.random.normal(loc=0.0, scale=np.sqrt(noise_var/2), size=np.shape(x_re))
        noise_im = np.random.normal(loc=0.0, scale=np.sqrt(noise_var/2), size=np.shape(x_im))

        return x_re + noise_re, x_im + noise_im


class source_relay_p2p_com:
    def __init__(self, dic_size, modulation_order, sig_pow=1.0):
        self.codewords = self.init_codewords(dic_size)
        # self.channel_encoder =
        # self.qpsk = QPSK(sig_pow)
        self.modulator = modulation(modulation_order)
        self.channel = AWGN()
        self.modulation_order = modulation_order

    def init_codewords(self, num_symbols):
        codewords = []
        self.len_codewords = math.ceil(np.log2(num_symbols))
        tot = 0
        for x in map(''.join, itertools.product('01', repeat=self.len_codewords)):
            if tot == num_symbols:
                break
            tot += 1
            codewords.append(x)
        return codewords

    def bit_error_rate_control(self, order, n_test, SNRs):
        test_modulator = modulation(order)
        bits = np.random.randint(0, 2, size=n_test)
        test_channel = AWGN()
        bits_per_symbol = np.log2(order)
        padding = int(len(bits) % bits_per_symbol)
        if not padding == 0:
            padding = int(bits_per_symbol - padding)
            zeros = np.zeros(padding)
            bits = np.concatenate((bits, zeros))
        modulated = test_modulator.modulate(bits)
        ch_in_re, ch_in_im = modulated[:, 0], modulated[:, 1]

        for SNR in SNRs:
            ch_out_re, ch_out_im = test_channel(ch_in_re, ch_in_im, SNR)
            bit_seq_hat = test_modulator.demodulate(ch_out_re, ch_out_im)
            print(f"Bit error rate: {1 - np.sum(bits == bit_seq_hat) / len(bits)}")
            linear_snr = np.power(10, SNR / 10)
            err_term = erfc(np.sqrt(3 * linear_snr / (2 * (order - 1))))
            print(f"Theoretical: {2 * err_term * (1 - (1 / np.sqrt(order))) / bits_per_symbol}")
            print("-" * 50)

    def communicate(self, token_indices, SNR):
        source_coded_bits = np.array(list("".join([self.codewords[i] for i in token_indices]))).astype(int)

        # channel_coded_bits = self.channel_encoder.encode(XXX)
        channel_coded_bits = source_coded_bits

        bits_per_symbol = np.log2(self.modulation_order)
        padding = int(len(channel_coded_bits) % bits_per_symbol)
        if not padding == 0:
            padding = int(bits_per_symbol - padding)
            zeros = np.zeros(padding)
            channel_coded_bits = np.concatenate((channel_coded_bits, zeros))

        modulated = self.modulator.modulate(channel_coded_bits)
        ch_in_re, ch_in_im = modulated[:, 0], modulated[:, 1]
        ch_out_re, ch_out_im = self.channel(ch_in_re, ch_in_im, SNR)
        bit_seq_hat = self.modulator.demodulate(ch_out_re, ch_out_im)
        if not padding == 0:
            bit_seq_hat = bit_seq_hat[:-padding]

        # channel_decoded_bits = self.channel_encoder.decode(bit_seq_hat)
        channel_decoded_bits = bit_seq_hat

        grouped_bit_seq = [channel_decoded_bits[i:i+self.len_codewords] for i in range(0, len(channel_decoded_bits), self.len_codewords)]
        codewords_matrix = np.reshape(list("".join(self.codewords)), (len(self.codewords), self.len_codewords)).astype(int)
        codewords_decoded = []
        for received_codeword in grouped_bit_seq:
            hamming_distance = np.sum(codewords_matrix != np.tile(received_codeword, (len(self.codewords), 1)), axis=1)
            codewords_decoded.append(np.argmin(hamming_distance))
        return codewords_decoded


def plotter(x_axis_values, y_axis_values, x_label, y_label, title):
    plt.figure()
    plt.plot(x_axis_values, y_axis_values)
    plt.grid()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(f"{title}.png", dpi=400)


def sbert_semantic_similarity_score(target_sentence, received_sentence, sbert_model):
    target_emb = sbert_model.encode(target_sentence, convert_to_tensor=True).unsqueeze(0)
    received_emb = sbert_model.encode(received_sentence, convert_to_tensor=True).unsqueeze(0)
    scores = F.cosine_similarity(target_emb, received_emb)
    return scores[0].item()


def semantic_similarity_score(target_sentences, received_sentences):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are skilled in evaluating how similar the two sentences are. Provide a number between -1 "
                "and 1 denoting the semantic similarity score for given sentences A and B with precision "
                "0.01. 1 means they are perfectly similar and -1 means they are opposite while 0 means their "
                "meanings are uncorrelated. Just provide a score without any words or symbols.",
            },
            {
                "role": "user",
                "content": f"A=({target_sentences})  B=({received_sentences})",
            },
        ],
    )

    if completion.choices[0].finish_reason == "stop":
        pattern = re.compile(r"(?<![\d.-])-?(?:0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])")
        res = pattern.findall(completion.choices[0].message.content)
        if len(res) == 1:
            return float(res[0])
        else:
            print(res)
            return float("nan")
    else:
        return float("nan")


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

    source_relay_comm = source_relay_p2p_com(data_handler.vocab_size, 256)

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
        snr_db = 10 * np.log10(1 / (np.power(d_sr, args.alpha) * args.noise_pow))

        for b in data_handler.test_dataloader:
            encoder_idx = b[0]
            encoder_attention_mask = b[1]
            encoder_idx = data_handler.label_encoder.transform(encoder_idx).cpu().numpy()


            predicted_ids = source_relay_comm.communicate(np.squeeze(encoder_idx), snr_db)

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

            sim_score = semantic_similarity_score(s1, s2)
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

            cosine_scores.append(sim_score)
            bleu1_scores.append(bleu_1_score)
            bleu_scores.append(bleu_score)
            sbert_semantic_sim_scores.append(sbert_sim_score)

            records.append([d_sr, s1, s2, sim_score, bleu_1_score, bleu_score, sbert_sim_score])

            if len(bleu1_scores) >= args.n_test:
                break

        n_test_samples = len(bleu1_scores)
        cosine_scores = [x for x in cosine_scores if not np.isnan(x)]

        mean_semantic_sim[distance_index, 0] = np.mean(cosine_scores)
        mean_sbert_semantic_sim[distance_index, 0] = np.mean(sbert_semantic_sim_scores)
        mean_bleu_1[distance_index, 0] = np.mean(bleu1_scores)
        mean_bleu[distance_index, 0] = np.mean(bleu_scores)

        std_semantic_sim[distance_index, 0] = np.std(
            cosine_scores, ddof=1
        ) / np.sqrt(n_test_samples)
        std_bleu_1[distance_index, 0] = np.std(bleu1_scores, ddof=1) / np.sqrt(
            n_test_samples
        )
        std_bleu[distance_index, 0] = np.std(bleu_scores, ddof=1) / np.sqrt(
            n_test_samples
        )

        std_sbert_semantic_sim[distance_index, 0] = np.std(sbert_semantic_sim_scores, ddof=1) / np.sqrt(
            n_test_samples)

        np.save("classic_conventional_mean_semantic_sim.npy", mean_semantic_sim)
        np.save("classic_conventional_mean_sbert_semantic_sim.npy", mean_sbert_semantic_sim)
        np.save("classic_conventional_mean_bleu_1.npy", mean_bleu_1)
        np.save("classic_conventional_mean_bleu.npy", mean_bleu)

        np.save("classic_conventional_std_semantic_sim.npy", std_semantic_sim)
        np.save("classic_conventional_std_sbert_semantic_sim.npy", std_sbert_semantic_sim)
        np.save("classic_conventional_std_bleu_1.npy", std_bleu_1)
        np.save("classic_conventional_std_bleu.npy", std_bleu)

        df = pd.DataFrame(
            records,
            columns=[
                "d_sr",
                "Sentence 1",
                "Sentence 2",
                "Semantic Similarity Score",
                "BLEU 1 Gram Score",
                "BLEU Score",
                "SBERT Semantic Score"
            ],
        )
        df.to_excel("classic_baseline_output.xlsx", index=False)
