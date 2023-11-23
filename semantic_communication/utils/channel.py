from abc import ABC
import torch
import numpy as np
from semantic_communication.utils.general import get_device


class Channel(ABC):
    def __init__(
        self,
        signal_power_constraint: float,
        alpha: float,
        noise_pow: float,
    ):
        self.signal_power_constraint = signal_power_constraint
        self.alpha = alpha
        self.noise_pow = noise_pow
        self.device = get_device()

    def __call__(self, x: torch.Tensor, d: float) -> torch.Tensor:
        pass

    def signal_process(self, x: torch.Tensor, d: float) -> torch.Tensor:
        # convert to complex
        last_dim = int(x.shape[-1] / 2)
        x = torch.complex(*torch.split(x, last_dim, dim=-1))

        # normalize
        sig_pow = self.signal_power_constraint / (d**self.alpha)
        x = (x / torch.abs(x)) * (sig_pow**0.5)

        return x


class AWGN(Channel):
    def __init__(self, signal_power_constraint, alpha, noise_pow):
        super().__init__(signal_power_constraint, alpha, noise_pow)

    def __call__(self, x: torch.Tensor, d: float) -> torch.Tensor:
        x = self.signal_process(x, d)

        noise = torch.normal(
            mean=0.0,
            std=self.noise_pow**0.5,
            size=x.shape,
            dtype=torch.cfloat,
        ).to(self.device)

        y = x + noise
        return torch.cat((y.real, y.imag), dim=-1)


class Rayleigh(Channel):
    def __init__(self, signal_power_constraint, alpha, noise_pow):
        super().__init__(signal_power_constraint, alpha, noise_pow)

    def __call__(self, x: torch.Tensor, d: float) -> torch.Tensor:
        B, T, C = x.shape
        x = self.signal_process(x)

        h_re = torch.div(torch.randn(B, T, device=self.device), 2**0.5)
        h_im = torch.div(torch.randn(B, T, device=self.device), 2**0.5)

        h_re = h_re.unsqueeze(2).repeat(1, 1, int(C / 2))
        h_im = h_im.unsqueeze(2).repeat(1, 1, int(C / 2))

        y = torch.zeros(x.shape).to(self.device)
        y[:, :, 0, :] = x[:, :, 0, :] * h_re - x[:, :, 1, :] * h_im
        y[:, :, 1, :] = x[:, :, 0, :] * h_im + x[:, :, 1, :] * h_re

        linear_SNR = np.power(10, SNR / 10)
        noise_var = self.signal_power_constraint / linear_SNR
        noise = torch.normal(
            mean=0.0,
            std=(noise_var / 2) ** 0.5,
            size=x.shape,
        ).to(self.device)

        y = y + noise
        return torch.reshape(y, (B, T, C))


def init_channel(
    channel_type: str,
    signal_power_constraint: float,
    alpha: float,
    noise_pow: float,
) -> Channel:
    if channel_type == "AWGN":
        return AWGN(signal_power_constraint, alpha, noise_pow)
    elif channel_type == "Rayleigh":
        return Rayleigh(signal_power_constraint, alpha, noise_pow)
    else:
        raise ValueError("Channel type should be AWGN or Rayleigh!")


def get_distance(d_min, d_max):
    return (torch.rand(1) * (d_max - d_min) + d_min).item()
