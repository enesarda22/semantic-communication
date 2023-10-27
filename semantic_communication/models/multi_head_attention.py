import torch
from torch import nn

from semantic_communication.models.self_attention_head import SelfAttentionHead


# TODO: add multiple heads as another dimension
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
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attention_mask):
        out = torch.cat([h(x, attention_mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))

        return out
