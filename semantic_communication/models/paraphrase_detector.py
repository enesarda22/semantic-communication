from typing import Optional, List

import torch
from torch import nn

from semantic_communication.models.semantic_encoder import SemanticEncoder


class ParaphraseDetector(nn.Module):
    def __init__(
        self,
        semantic_encoder: SemanticEncoder,
        n_in: int = 384,
        n_latent: int = 512,
    ):
        super().__init__()
        self.semantic_encoder = semantic_encoder

        self.fc1 = nn.Linear(n_in, n_latent)
        self.fc2 = nn.Linear(n_latent, n_latent)
        self.fc3 = nn.Linear(n_latent, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        messages: Optional[List[str]] = None,
        m1: Optional[List[str]] = None,
        m2: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        x = self.semantic_encoder(
            messages=messages,
            m1=m1,
            m2=m2,
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).squeeze(1)
        x = self.relu(self.fc1(x))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x).flatten()
