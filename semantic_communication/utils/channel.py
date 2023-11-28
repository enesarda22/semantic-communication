from abc import ABC
import torch
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

    def signal_process(self, x: torch.Tensor, d: float, h=None) -> torch.Tensor:
        # convert to complex
        last_dim = int(x.shape[-1] / 2)
        x = torch.complex(*torch.split(x, last_dim, dim=-1))

        if not h is None:
            x = x / h

        # normalize
        sig_pow = self.signal_power_constraint / (d ** self.alpha)
        x = (x / torch.abs(x)) * (sig_pow ** 0.5)

        return x


class AWGN(Channel):
    def __init__(self, signal_power_constraint, alpha, noise_pow):
        super().__init__(signal_power_constraint, alpha, noise_pow)

    def __call__(self, x: torch.Tensor, d: float) -> torch.Tensor:
        x = self.signal_process(x, d)

        noise = torch.normal(
            mean=0.0,
            std=self.noise_pow ** 0.5,
            size=x.shape,
            dtype=torch.cfloat,
        ).to(self.device)

        y = x + noise
        return torch.cat((y.real, y.imag), dim=-1)


class Rayleigh(Channel):
    def __init__(self, signal_power_constraint, alpha, noise_pow):
        super().__init__(signal_power_constraint, alpha, noise_pow)

    def __call__(self, x: torch.Tensor, d: float) -> torch.Tensor:
        h = torch.normal(
            mean=0.0,
            std=1.0,
            size=x.shape,
            dtype=torch.float,
        ).to(self.device)

        last_dim = int(h.shape[-1] / 2)
        h = torch.complex(*torch.split(h, last_dim, dim=-1))

        x = self.signal_process(x, d, h)

        noise = torch.normal(
            mean=0.0,
            std=self.noise_pow ** 0.5,
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
) -> Channel:
    if channel_type == "AWGN":
        return AWGN(signal_power_constraint, alpha, noise_pow)
    elif channel_type == "Rayleigh":
        return Rayleigh(signal_power_constraint, alpha, noise_pow)
    else:
        raise ValueError("Channel type should be AWGN or Rayleigh!")


def get_distance(d_min, d_max):
    return (torch.rand(1) * (d_max - d_min) + d_min).item()
