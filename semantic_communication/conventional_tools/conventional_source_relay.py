import numpy as np
from semantic_communication.conventional_tools.bit_reed_solomon import BitReedSolomon
from semantic_communication.utils.modulation import modulation
from semantic_communication.conventional_tools.conv_channels import AWGN
from scipy.special import erfc
import math
import itertools


class source_relay_p2p_com:
    def __init__(self, dic_size, modulation_order, channel_code=False, sig_pow=1.0):
        self.len_codewords = 0
        self.codewords = self.init_codewords(dic_size)
        if channel_code:
            self.channel_encoder = BitReedSolomon(n=31, k=29, m=5)

        self.channel_code = channel_code

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

        if self.channel_code:
            channel_coded_bits = (self.channel_encoder.encode_bit_sequence(source_coded_bits)).astype(int)
        else:
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

        if self.channel_code:
            channel_decoded_bits = (self.channel_encoder.decode_bit_sequence(bit_seq_hat.astype(int))).astype(int)
        else:
            channel_decoded_bits = bit_seq_hat

        grouped_bit_seq = [channel_decoded_bits[i:i+self.len_codewords] for i in range(0, len(channel_decoded_bits), self.len_codewords)]
        codewords_matrix = np.reshape(list("".join(self.codewords)), (len(self.codewords), self.len_codewords)).astype(int)
        codewords_decoded = []
        for received_codeword in grouped_bit_seq:
            hamming_distance = np.sum(codewords_matrix != np.tile(received_codeword, (len(self.codewords), 1)), axis=1)
            codewords_decoded.append(np.argmin(hamming_distance))
        return codewords_decoded