import torch
from torch import nn

from semantic_communication.model.self_attention_head import SelfAttentionHead


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, embedding_size, head_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                SelfAttentionHead(
                    embedding_size=embedding_size,
                    head_size=head_size,
                    block_size=block_size,
                )
                for _ in range(n_heads)
            ]
        )
        self.proj = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)

        return out
