import numpy as np
from abc import ABC


class conv_channel(ABC):
    def __init__(
        self,
        signal_power_constraint: float,
        alpha: float,
        noise_pow: float,
    ):
        self.signal_power_constraint = signal_power_constraint
        self.alpha = alpha
        self.noise_pow = noise_pow


class conv_AWGN(conv_channel):
    def __init__(self, signal_power_constraint, alpha, noise_pow):
        super().__init__(signal_power_constraint, alpha, noise_pow)

    def __call__(self, x_re, x_im, d):
        noise_var = self.noise_pow * (d**self.alpha)
        noise_re, noise_im = np.random.normal(loc=0.0, scale=np.sqrt(noise_var / 2), size=(2, len(x_re)))

        return x_re + noise_re, x_im + noise_im


class conv_Rayleigh(conv_channel):
    def __init__(self, signal_power_constraint, alpha, noise_pow):
        super().__init__(signal_power_constraint, alpha, noise_pow)

    def __call__(self, x_re, x_im, d):
        noise_var = self.noise_pow * (d**self.alpha)
        noise_re, noise_im = np.random.normal(loc=0.0, scale=np.sqrt(noise_var/2), size=(2, len(x_re)))

        h_re, h_im = np.random.normal(loc=0.0, scale=np.sqrt(1/2), size=2)

        complex = x_re + 1j * x_im
        h = h_re + 1j * h_im
        complex_noise = noise_re + 1j * noise_im

        # Transmit power control
        processed_complex = (np.conjugate(h) * complex / np.abs(h))

        y = h * processed_complex + complex_noise
        return np.ascontiguousarray(y.real), np.ascontiguousarray(y.imag)





