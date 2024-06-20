import numpy as np
from semantic_communication.conventional_tools.bit_reed_solomon import BitReedSolomon
from semantic_communication.utils.modulation import modulation
from semantic_communication.conventional_tools.conv_channels import conv_AWGN, conv_Rayleigh
from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.conventional_tools.fixed_length_coding import FixedLengthCoding, pad_encoded_sequence
from scipy.optimize import curve_fit
from scipy.stats import norm
import os
from tqdm import tqdm
from semantic_communication.utils.general import get_device


class ConventionalTransmitter:
    def __init__(self, source_coding, modulation_block: modulation, channel_coding=None):
        self.source_coding = source_coding
        self.channel_coding = channel_coding
        self.modulation = modulation_block
        self.m = 0
        if not channel_coding is None:
            self.m = channel_coding.m

    def __call__(self, x):
        if not self.channel_coding is None:
            source_coded, source_extra_padding = self.source_coding.encode(x, self.m)
            channel_coded = self.channel_coding.encode_bit_sequence(source_coded)
            channel_coded, channel_extra_padding = pad_encoded_sequence(channel_coded, int(np.log2(self.modulation.order)))
            modulated = self.modulation.modulate(channel_coded)
        else:
            source_coded, channel_extra_padding = self.source_coding.encode(x, int(np.log2(self.modulation.order)))
            modulated = self.modulation.modulate(source_coded)
            source_extra_padding = 0

        return modulated[:, 0], modulated[:, 1], source_extra_padding, channel_extra_padding


class ConventionalReceiver:
    def __init__(self, source_coding, modulation_block: modulation, channel_coding=None):
        self.source_coding = source_coding
        self.channel_coding = channel_coding
        self.modulation = modulation_block
        self.m = 0
        if not channel_coding is None:
            self.m = channel_coding.m

    def __call__(self, ch_out_re, ch_out_im, src_padding, ch_padding):
        demodulated = self.modulation.demodulate(ch_out_re, ch_out_im)
        if not ch_padding == 0:
            demodulated = demodulated[:-ch_padding]

        if not self.channel_coding is None:
            channel_decoded = self.channel_coding.decode_bit_sequence(demodulated.astype(int))
            if not src_padding == 0:
                channel_decoded = channel_decoded[:-src_padding]
            source_decoded = self.source_coding.decode(channel_decoded)
        else:
            source_decoded = self.source_coding.decode(demodulated)
        return source_decoded


class conventional_three_node_network:
    def __init__(self, data_handler: DataHandler, channel_coding: bool, channel_type: str, sig_pow, alpha, noise_pow, d_grid, train_transition=True, n_train=50000, data_fp="Data"):
        self.source_coding = FixedLengthCoding(data_handler.vocab_size)
        self.data_handler = data_handler
        self.device = get_device()

        if channel_coding:
            self.modulation_block = modulation(512)
            self.channel_coding = BitReedSolomon(7, 5, 3)
            self.modulation_order = 512

        else:
            self.modulation_block = modulation(256)
            self.channel_coding = None
            self.modulation_order = 256

        print(f"Modulation order: {self.modulation_order} || Channel coding: {channel_coding}")

        if channel_type == "AWGN":
            self.channel = conv_AWGN(signal_power_constraint=sig_pow, alpha=alpha, noise_pow=noise_pow)
        elif channel_type == "Rayleigh":
            self.channel = conv_Rayleigh(signal_power_constraint=sig_pow, alpha=alpha, noise_pow=noise_pow)
        else:
            raise ValueError("Invalid channel type.")

        self.source_transmitter = ConventionalTransmitter(source_coding=self.source_coding, modulation_block=self.modulation_block, channel_coding=self.channel_coding)
        self.relay_receiver = ConventionalReceiver(source_coding=self.source_coding, modulation_block=self.modulation_block, channel_coding=self.channel_coding)
        self.relay_transmitter = ConventionalTransmitter(source_coding=self.source_coding, modulation_block=self.modulation_block, channel_coding=self.channel_coding)
        self.destination_receiver = ConventionalReceiver(source_coding=self.source_coding, modulation_block=self.modulation_block, channel_coding=self.channel_coding)

        self.data_fp = data_fp

        if train_transition:
            self.train_probability(n_train, d_grid)

        p_transition = np.load(os.path.join(data_fp, "p_transition.npy"))
        p_transition = p_transition.reshape(-1, self.modulation_order, self.modulation_order)

        self.logistic_params = self._fit_probabilities(
            distances=d_grid,
            p_transition=p_transition,
            modulation_order=self.modulation_order
        )

    def __call__(self, x, d_sd, d_sr, d_rd):
        s_out_re, s_out_im, s_s_padding, s_ch_padding = self.source_transmitter(x)
        r_in_re, r_in_im = self.channel(s_out_re, s_out_im, d_sr)
        r_decoded = self.relay_receiver(r_in_re, r_in_im, s_s_padding, s_ch_padding)

        r_out_re, r_out_im, r_s_padding, r_ch_padding = self.relay_transmitter(r_decoded)
        dr_in_re, dr_in_im = self.channel(r_out_re, r_out_im, d_rd)
        ds_in_re, ds_in_im = self.channel(s_out_re, s_out_im, d_sd)

        d_in = self.ml_decision(ds_in_re, ds_in_im, dr_in_re, dr_in_im, d_sd, d_sr, d_rd)
        d_in_re, d_in_im = np.split(d_in, 2, axis=-1)
        return self.destination_receiver(d_in_re, d_in_im, s_s_padding, s_ch_padding)

    def ml_decision(self, ds_in_real, ds_in_imag, dr_in_real, dr_in_imag, d_sd, d_sr, d_rd):
        relay_transition_log_prob = self.get_relay_transition_log_prob(d_sr)
        log_likelihoods = np.empty((ds_in_real.shape[0], self.modulation_order))
        modulation_codebook = self.modulation_block.get_codebook()
        for i, s_code in enumerate(modulation_codebook):
            sd_log_likelihood = self.log_likelihood(
                x=[ds_in_real, ds_in_imag],
                mean=[s_code.real, s_code.imag],
                d=d_sd,
            )

            rd_log_likelihood = np.full(dr_in_real.shape, -np.inf)

            for j, r_code in enumerate(modulation_codebook):
                log_likelihood = self.log_likelihood(
                    x=[dr_in_real, dr_in_imag],
                    mean=[r_code.real, r_code.imag],
                    d=d_rd,
                )
                rd_log_likelihood = np.logaddexp(
                    rd_log_likelihood,
                    relay_transition_log_prob[i, j] + log_likelihood,
                )

            log_likelihoods[:, i] = sd_log_likelihood + rd_log_likelihood

        max_indices = np.argmax(log_likelihoods, axis=1)
        ml_codes = modulation_codebook[max_indices]
        return np.concatenate([ml_codes.real, ml_codes.imag])

    def log_likelihood(self, x, mean, d):
        return norm.logpdf(
            x=x[0],
            loc=mean[0],
            scale=(self.channel.noise_pow * (d ** self.channel.alpha) / 2) ** 0.5,
        ) + norm.logpdf(
            x=x[1],
            loc=mean[1],
            scale=(self.channel.noise_pow * (d ** self.channel.alpha) / 2) ** 0.5,
        )

    def get_relay_transition_log_prob(self, d_sr):
        transition_log_proba = np.empty((self.modulation_order, self.modulation_order))
        for i in range(self.modulation_order):
            for j in range(self.modulation_order):
                transition_log_proba[i, j] = self.log_logistic(
                    x=d_sr,
                    L=self.logistic_params[i, j, 0],
                    k=self.logistic_params[i, j, 1],
                    x0=self.logistic_params[i, j, 2],
                )

        return transition_log_proba

    @classmethod
    def _fit_probabilities(cls, distances, p_transition, modulation_order):
        params = np.empty((modulation_order, modulation_order, 3))
        for i in range(modulation_order):
            for j in range(modulation_order):
                popt, _ = curve_fit(
                    f=cls.logistic,
                    xdata=distances,
                    ydata=p_transition[:, i, j],
                    p0=[0.25, 1, 3500],
                    method="trf",
                )
                params[i, j] = popt

        return params

    @staticmethod
    def logistic(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    @staticmethod
    def log_logistic(x, L, k, x0):
        return np.log(L) - np.log1p(np.exp(-k * (x - x0)))

    def train_probability(self, n_train, d_grid):
        p_transitions = []
        modulation_codebook = self.modulation_block.get_codebook()
        for d in d_grid:
            print(f"Iteration for d: {d}")
            p_transition = np.zeros((self.modulation_order, self.modulation_order))
            i = 0
            for b in tqdm(self.data_handler.train_dataloader):
                mask = (b[0] != 0) & (b[0] != 101) & (b[0] != 102)
                b_input_ids = b[0].masked_select(mask)
                b_input_ids = self.data_handler.label_encoder.transform(b_input_ids).cpu().detach().numpy()

                source_out_re, source_out_im, s_s_padding, s_ch_padding = self.source_transmitter(b_input_ids)

                r_in_re, r_in_im = self.channel(source_out_re, source_out_im, d)
                r_decoded = self.relay_receiver(r_in_re, r_in_im, s_s_padding, s_ch_padding)

                r_out_re, r_out_im, r_s_padding, r_ch_padding = self.relay_transmitter(r_decoded)
                modulated_s = source_out_re + 1j * source_out_im
                modulated_r = r_out_re + 1j * r_out_im

                for k, code_s in enumerate(modulation_codebook):
                    temp_modulated_r = modulated_r[modulated_s == code_s]
                    for l, code_r in enumerate(modulation_codebook):
                        p_transition[k, l] += np.sum(temp_modulated_r == code_r)

                i += 1
                if i >= n_train:
                    break
            p_transition = p_transition / p_transition.sum(axis=1, keepdims=True)
            p_transitions.append(p_transition)

        np.save(os.path.join(self.data_fp, "distances.npy"), d_grid)
        np.save(os.path.join(self.data_fp, "p_transition.npy"), np.concatenate(p_transitions))
