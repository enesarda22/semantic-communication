import torch
from torch import nn
from torch.nn import functional as F


class SelfAttentionHead(nn.Module):
    def __init__(self, embedding_size, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attention_mask):
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        v = self.value(x)  # (B,T,C)

        # TODO: allow tokens to communicate with future tokens
        wei = q @ k.transpose(-2, -1) * (v.shape[2] ** -0.5)  # (B,T,T)
        wei = wei.masked_fill(self.tril == 0, -torch.inf)  # (B,T,T)

        extended_mask = attention_mask.unsqueeze(-1)
        extended_mask = torch._cast_Double(extended_mask)  @ torch._cast_Double(extended_mask.transpose(1, 2))
        wei = wei.masked_fill(extended_mask == 0, -torch.inf)

        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        wei = torch.nan_to_num(wei)

        wei = self.dropout(wei)
        out = wei @ v  # (B,T,C)
        return out
