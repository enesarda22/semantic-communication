import warnings
from abc import ABC
from typing import Optional

import torch
import torch.nn.functional as F

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

    def __call__(
        self,
        x: torch.Tensor,
        d: Optional[float] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pass

    def signal_process(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        assert C % 2 == 0

        # normalize
        x = self.normalize(x, self.signal_power_constraint, attention_mask)

        # convert to complex
        x = torch.complex(*torch.split(x, int(C / 2), dim=-1))
        return x

    @staticmethod
    def normalize(x, pow, attention_mask: Optional[torch.Tensor] = None):
        B, T, C = x.shape

        if attention_mask is None:
            x = ((pow * T * C / 2) ** 0.5) * F.normalize(
                x.reshape(B, T * C), dim=-1, p=2
            ).reshape(B, T, C)
        else:
            for i in range(B):
                n_tokens = int(torch.sum(attention_mask[i, :]).item())
                x[i, :n_tokens, :] = ((pow * n_tokens * C / 2) ** 0.5) * F.normalize(
                    x[i, :n_tokens, :].flatten(), dim=-1, p=2
                ).reshape(n_tokens, C)
        return x


class AWGN(Channel):
    def __init__(self, signal_power_constraint, alpha, noise_pow):
        super().__init__(signal_power_constraint, alpha, noise_pow)

    def __call__(
        self,
        x: torch.Tensor,
        d: Optional[float] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if d is None:
            return x

        x = self.signal_process(x, attention_mask=attention_mask)

        noise = torch.normal(
            mean=0.0,
            std=(self.noise_pow * (d**self.alpha)) ** 0.5,  # TODO: path loss
            size=x.shape,
            dtype=torch.cfloat,
        ).to(self.device)

        y = x + noise
        return torch.cat((y.real, y.imag), dim=-1)


class Rayleigh(Channel):
    def __init__(self, signal_power_constraint, alpha, noise_pow):
        super().__init__(signal_power_constraint, alpha, noise_pow)

    def __call__(
        self,
        x: torch.Tensor,
        d: Optional[float] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if d is None:
            return x

        h = torch.normal(
            mean=0.0,
            std=1.0,
            size=(*x.shape[:-1], 1),
            dtype=torch.cfloat,
        ).to(self.device)

        x = self.signal_process(x, attention_mask)
        x = x * torch.conj(h) / torch.abs(h)

        noise = torch.normal(
            mean=0.0,
            std=(self.noise_pow * (d**self.alpha)) ** 0.5,
            size=x.shape,
            dtype=torch.cfloat,
        ).to(self.device)

        y = h * x + noise
        return torch.cat((y.real, y.imag), dim=-1)


def init_channel(
    channel_type: str,
    signal_power_constraint: float,
    alpha: float,
    noise_pow: float,
) -> Optional[Channel]:
    if channel_type == "AWGN":
        return AWGN(signal_power_constraint, alpha, noise_pow)
    elif channel_type == "Rayleigh":
        return Rayleigh(signal_power_constraint, alpha, noise_pow)
    else:
        warnings.warn("Channel is None!")
        return None


def get_distance(d_min, d_max):
    return (torch.rand(1) * (d_max - d_min) + d_min).item()
