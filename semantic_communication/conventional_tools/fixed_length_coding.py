import numpy as np
import math
import itertools


def pad_encoded_sequence(encoded_sequence, m):
    mod = encoded_sequence.shape[0] % m
    if mod == 0:
        extra_padding = 0
    else:
        extra_padding = m - mod

    encoded_sequence = np.append(
        encoded_sequence,
        np.zeros(extra_padding, dtype=np.uint8),
    )
    return encoded_sequence, extra_padding


class FixedLengthCoding:
    def __init__(self, dic_size):
        self.m = 0
        self.codewords = self.init_codewords(dic_size)

    def init_codewords(self, len_codebook):
        codewords = []
        len_codewords = math.ceil(np.log2(len_codebook))
        self.m = len_codewords
        tot = 0
        for x in map(''.join, itertools.product('01', repeat=len_codewords)):
            if tot == len_codebook:
                break
            tot += 1
            codewords.append(x)
        return codewords

    def encode(self, tokens, m):
        bit_sequence = np.array(list("".join([self.codewords[i] for i in tokens]))).astype(int)
        padded_sequence, extra_padding = pad_encoded_sequence(
            encoded_sequence=bit_sequence,
            m=m,
        )

        return padded_sequence, extra_padding

    def decode(self, encoded_sequence):
        grouped_bit_seq = [encoded_sequence[i:i+self.m] for i in range(0, len(encoded_sequence), self.m)]
        codewords_matrix = np.reshape(list("".join(self.codewords)), (len(self.codewords), self.m)).astype(int)
        codewords_decoded = []
        for received_codeword in grouped_bit_seq:
            hamming_distance = np.sum(codewords_matrix != np.tile(received_codeword, (len(self.codewords), 1)), axis=1)
            codewords_decoded.append(np.argmin(hamming_distance))
        return codewords_decoded

    @staticmethod
    def pad_zeros(encoded_sequence, m):
        mod = encoded_sequence.shape[0] % m
        if mod == 0:
            extra_padding = 0
        else:
            extra_padding = m - mod

        return np.append(
            encoded_sequence,
            np.zeros(extra_padding, dtype=np.uint8),
        )

