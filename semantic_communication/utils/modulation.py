import numpy as np
import matlab.engine


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
