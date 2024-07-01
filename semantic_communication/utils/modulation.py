import numpy as np
import matlab.engine
import math
import itertools


class modulation:
    def __init__(self, order):
        self.order = order
        self.eng = matlab.engine.start_matlab()

    def modulate(self, bit_sequence):
        modulated = self.eng.modulator(np.expand_dims(bit_sequence.astype(float), axis=1), float(self.order))
        return np.array(modulated)

    def demodulate(self, ch_out_re, ch_out_im):
        demodulated = np.squeeze(self.eng.demodulator(ch_out_re, ch_out_im, float(self.order)))
        return np.squeeze(demodulated.reshape(-1, 1, order='F'))

    def init_codewords(self):
        codewords = []
        len_codewords = math.ceil(np.log2(self.order))
        tot = 0
        for x in map(''.join, itertools.product('01', repeat=len_codewords)):
            if tot == self.order:
                break
            tot += 1
            codewords.append(x)
        return codewords

    def get_codebook(self):
        tmp_codewords = self.init_codewords()
        bits = np.array(list("".join([tmp_codewords[i] for i in range(self.order)]))).astype(int)
        alphabet = self.modulate(bits)
        return alphabet[:, 0] + 1j * alphabet[:, 1]



