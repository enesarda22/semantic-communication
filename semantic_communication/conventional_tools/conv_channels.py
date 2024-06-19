import numpy as np


class AWGN:
    def __init__(self, signal_power_constraint=1.0):
        self.signal_power_constraint = signal_power_constraint

    def __call__(self, x_re, x_im, SNR):
        linear_SNR = np.power(10, SNR / 10)
        noise_var = self.signal_power_constraint / linear_SNR
        noise_re, noise_im = np.random.normal(loc=0.0, scale=np.sqrt(noise_var / 2), size=(2, len(x_re)))

        return x_re + noise_re, x_im + noise_im


class Rayleigh:
    def __init__(self, signal_power_constraint=1.0):
        self.signal_power_constraint = signal_power_constraint

    def __call__(self, x_re, x_im, SNR):
        linear_SNR = np.power(10, SNR / 10)
        noise_var = self.signal_power_constraint / linear_SNR
        noise_re, noise_im = np.random.normal(loc=0.0, scale=np.sqrt(noise_var/2), size=(2, len(x_re)))

        h_re, h_im = np.random.normal(loc=0.0, scale=np.sqrt(1/2), size=2)

        complex = x_re + 1j * x_im
        h = h_re + 1j * h_im
        complex_noise = noise_re + 1j * noise_im

        # Transmit power control
        processed_complex = (np.conjugate(h) * complex / np.abs(h))

        y = h * processed_complex + complex_noise
        return np.ascontiguousarray(y.real), np.ascontiguousarray(y.imag)

