from abc import ABC

import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Channel(ABC):
    def __init__(self, SNR: float, signal_power_constraint: float = 1):
        self.SNR = SNR
        self.linear_SNR = np.power(10, SNR / 10)
        self.noise_var = signal_power_constraint / self.linear_SNR
        self.signal_power_constraint = signal_power_constraint

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def signal_process(self, x):  # x.shape = (Batch, 2B)

        # Average power constraint, normalize to signal power constraint
        x = ((self.signal_power_constraint * x.size(dim=1) / 2)**0.5) * torch.nn.functional.normalize(x, dim=1, p=2)

        # Transform to complex (Batch, 2B) -> (Batch, 2, B)
        dim1, dim2 = x.shape
        n_d = int(dim2 / 2)
        x = torch.reshape(x, (dim1, 2, n_d))

        return x


class AWGN(Channel):
    def __init__(self, SNR, signal_power_constraint):
        super().__init__(SNR, signal_power_constraint)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:  # x.shape = (Batch, 2, B)
        init_dim1, init_dim2 = x.shape
        x = self.signal_process(x)
        y = x + torch.normal(mean=0.0, std=(self.noise_var / 2) ** 0.5, size=x.shape).to(device)
        return torch.reshape(y, (init_dim1, init_dim2))


class Rayleigh(Channel):
    def __init__(self, SNR, signal_power_constraint):
        super().__init__(SNR, signal_power_constraint)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:  # x.shape = (Batch, 2, B)
        init_dim1, init_dim2 = x.shape
        x = self.signal_process(x)

        h_re = (torch.randn(init_dim1, 1) / (2 ** 0.5)).to(device)
        h_im = (torch.randn(init_dim1, 1) / (2 ** 0.5)).to(device)

        y = torch.zeros(x.shape).to(device)

        y[:, 0, :] = x[:, 0, :] * h_re - x[:, 1, :] * h_im
        y[:, 1, :] = x[:, 0, :] * h_im + x[:, 1, :] * h_re
        y = y + torch.normal(mean=0.0, std=(self.noise_var / 2) ** 0.5, size=x.shape).to(device)

        return torch.reshape(y, (init_dim1, init_dim2))
