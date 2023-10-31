from abc import ABC

import torch
import numpy as np

import torch.nn.functional as F

from utils.general import get_device


class Channel(ABC):
    def __init__(self, SNR: float, signal_power_constraint: float = 1.0):
        self.SNR = SNR
        self.linear_SNR = np.power(10, SNR / 10)
        self.noise_var = signal_power_constraint / self.linear_SNR
        self.signal_power_constraint = signal_power_constraint
        self.device = get_device()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def signal_process(self, x: torch.Tensor):
        B, T, C = x.shape

        # Average power constraint, normalize to signal power constraint
        x = ((self.signal_power_constraint * C / 2) ** 0.5) * F.normalize(
            x, dim=2, p=2
        )

        # Transform to complex (Batch, 2B) -> (Batch, 2, B)
        n_d = int(C / 2)
        x = torch.reshape(x, (B, T, 2, n_d))
        return x


class AWGN(Channel):
    def __init__(self, SNR, signal_power_constraint):
        super().__init__(SNR, signal_power_constraint)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x = self.signal_process(x)
        noise = torch.normal(
            mean=0.0,
            std=(self.noise_var / 2) ** 0.5,
            size=x.shape,
        ).to(self.device)

        y = x + noise
        return torch.reshape(y, (B, T, C))


class Rayleigh(Channel):
    def __init__(self, SNR, signal_power_constraint):
        super().__init__(SNR, signal_power_constraint)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x = self.signal_process(x)

        h_re = torch.div(torch.randn(B, T, device=self.device), 2**0.5)
        h_im = torch.div(torch.randn(B, T, device=self.device), 2**0.5)

        h_re = h_re.unsqueeze(2).repeat(1, 1, int(C / 2))
        h_im = h_im.unsqueeze(2).repeat(1, 1, int(C / 2))

        y = torch.zeros(x.shape).to(self.device)
        y[:, :, 0, :] = x[:, :, 0, :] * h_re - x[:, :, 1, :] * h_im
        y[:, :, 1, :] = x[:, :, 0, :] * h_im + x[:, :, 1, :] * h_re

        y = y + torch.normal(
            mean=0.0,
            std=(self.noise_var / 2) ** 0.5,
            size=x.shape,
        ).to(self.device)

        return torch.reshape(y, (B, T, C))