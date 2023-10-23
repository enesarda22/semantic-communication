from abc import ABC

import torch


class Channel(ABC):
    def __init__(self, SNR: float, signal_power_constraint: float = 1):
        self.SNR = SNR
        self.signal_power_constraint = signal_power_constraint

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass
