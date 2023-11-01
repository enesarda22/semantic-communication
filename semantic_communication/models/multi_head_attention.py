import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, embedding_size, head_size, block_size):
        super().__init__()
        self.N = n_heads

        self.key = nn.Linear(embedding_size, head_size * n_heads, bias=False)
        self.query = nn.Linear(embedding_size, head_size * n_heads, bias=False)
        self.value = nn.Linear(embedding_size, head_size * n_heads, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )

        self.proj = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attention_mask):
        B, T, C = x.shape

        K = self.key(x).view(B, T, self.N, -1)
        Q = self.query(x).view(B, T, self.N, -1)
        V = self.value(x).view(B, T, self.N, -1)

        wei = torch.einsum("b i h d , b j h d -> b h i j", Q, K)
        wei = wei * (C**-0.5)  # normalize

        # TODO: allow tokens to communicate with future tokens
        # tril mask to disable communication with future tokens
        wei = wei.masked_fill(self.tril == 0, -torch.inf)  # (B,N,T,T)
        wei = wei.transpose(0, 1)  # (N,B,T,T)

        # attention mask to disable communication with paddings
        extended_mask = attention_mask.unsqueeze(-1).to(torch.float64)
        extended_mask = extended_mask @ extended_mask.transpose(1, 2)

        wei = F.softmax(wei, dim=-1)  # (N,B,T,T)
        wei.masked_fill(extended_mask == 0, 0)

        wei = self.dropout(wei)
        out = torch.einsum("h b j i, b i h d -> b j h d", wei, V)

        out = out.reshape(B, T, C)
        out = self.dropout(self.proj(out))
        return out
