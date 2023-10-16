import torch
import torch.nn as nn
from torch.nn import functional as F

from semantic_communication.model.multi_head_attention import MultiHeadAttention


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_heads, n_embeddings, block_size, device):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embeddings)
        self.position_embedding_table = nn.Embedding(block_size, n_embeddings)

        self.sa_heads = MultiHeadAttention(
            n_heads=n_heads,
            embedding_size=n_embeddings,
            head_size=n_embeddings // n_heads,
            block_size=block_size,
        )
        self.ff_net = nn.Sequential(
            nn.Linear(n_embeddings, n_embeddings),
            nn.ReLU(),
        )
        self.lm_head = nn.Linear(n_embeddings, vocab_size)

        self.block_size = block_size
        self.device = device

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx)  # (B,T,C)
        pos_embeddings = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T,C)
        x = token_embeddings + pos_embeddings
        x = self.sa_heads(x)
        x = self.ff_net(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.reshape(B * T, C)
            targets = targets.reshape(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

    def generate_from_scratch(self):
        idx = torch.ones((1, self.block_size), dtype=torch.long)
        for i in range(self.block_size - 1):
            # get the predictions
            logits, loss = self(idx)
            # generate new token
            logits = logits[:, i, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx[0, i + 1] = idx_next

        return idx
