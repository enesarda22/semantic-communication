import numpy as np
from abc import ABC
import math


def calculate_channel_symbol_lengths(valid_length, valid_counts, scale=15/8):
    values = np.array(valid_counts) * scale
    values = [math.floor(num) for num in values]
    total = np.sum(values)
    diff = valid_length - total
    values[-1] += diff
    return values


class conv_channel(ABC):
    def __init__(
        self,
        signal_power_constraint: float,
        alpha: float,
        noise_pow: float,
        p2p: bool
    ):
        self.signal_power_constraint = signal_power_constraint
        self.alpha = alpha
        self.noise_pow = noise_pow
        self.p2p = p2p


class conv_AWGN(conv_channel):
    def __init__(self, signal_power_constraint, alpha, noise_pow, p2p=False):
        super().__init__(signal_power_constraint, alpha, noise_pow, p2p)

    def __call__(self, x_re, x_im, d, valid_counts=0):
        noise_var = self.noise_pow * (d**self.alpha)
        noise_re, noise_im = np.random.normal(loc=0.0, scale=np.sqrt(noise_var / 2), size=(2, len(x_re)))

        return x_re + noise_re, x_im + noise_im


class conv_Rayleigh(conv_channel):
    def __init__(self, signal_power_constraint, alpha, noise_pow, p2p=False):
        super().__init__(signal_power_constraint, alpha, noise_pow, p2p)

    def __call__(self, x_re, x_im, d, valid_counts=0):

        if self.p2p:
            noise_var = self.noise_pow * (d ** self.alpha)
            noise_re, noise_im = np.random.normal(loc=0.0, scale=np.sqrt(noise_var / 2), size=(2, len(x_re)))

            h_re, h_im = np.random.normal(loc=0.0, scale=np.sqrt(1 / 2), size=2)

            complex = x_re + 1j * x_im
            h = h_re + 1j * h_im
            complex_noise = noise_re + 1j * noise_im

            # Transmit power control
            processed_complex = (np.conjugate(h) * complex / np.abs(h))

            y = h * processed_complex + complex_noise
            return np.ascontiguousarray(y.real), np.ascontiguousarray(y.imag)

        else:
            valid_counts = calculate_channel_symbol_lengths(len(x_re), valid_counts)
            noise_var = self.noise_pow * (d**self.alpha)
            noise_re, noise_im = np.random.normal(loc=0.0, scale=np.sqrt(noise_var/2), size=(2, len(x_re)))

            h = np.random.normal(loc=0.0, scale=np.sqrt(1/2), size=len(valid_counts)) + 1j * np.random.normal(loc=0.0, scale=np.sqrt(1/2), size=len(valid_counts))

            h = np.concatenate([np.full(length, h_temp) for h_temp, length in zip(h, valid_counts)])

            complex = x_re + 1j * x_im
            complex_noise = noise_re + 1j * noise_im

            # Transmit power control
            processed_complex = (np.conjugate(h) * complex / np.abs(h))

            y = h * processed_complex + complex_noise
            return np.ascontiguousarray(y.real), np.ascontiguousarray(y.imag)





