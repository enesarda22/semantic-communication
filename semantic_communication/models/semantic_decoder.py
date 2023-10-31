import torch
import torch.nn as nn
from torch.nn import functional as F

from semantic_communication.models.multi_head_attention import (
    MultiHeadAttention,
)

from semantic_communication.utils.general import get_device


class SemanticDecoder(nn.Module):
    def __init__(self, vocab_size, n_heads, n_embeddings, block_size):
        super().__init__()
        self.sa_heads = MultiHeadAttention(
            n_heads=n_heads,
            embedding_size=n_embeddings,
            head_size=n_embeddings // n_heads,
            block_size=block_size,
        )
        self.ff_net = nn.Sequential(
            nn.Linear(n_embeddings, 4 * n_embeddings),
            nn.ReLU(),
            nn.Linear(4 * n_embeddings, n_embeddings),  # projection
            nn.Dropout(0.1),
        )
        self.ln1 = nn.LayerNorm(n_embeddings)
        self.ln2 = nn.LayerNorm(n_embeddings)
        self.ln3 = nn.LayerNorm(n_embeddings)
        self.lm_head = nn.Linear(n_embeddings, vocab_size)

        self.block_size = block_size

    def forward(self, encoder_output, attention_mask=None, targets=None):
        B, T, C = encoder_output.shape

        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.long).to(get_device())

        # residual connection after the layer, norm before the layer
        x = encoder_output + self.sa_heads(
            self.ln1(encoder_output), attention_mask
        )
        x = x + self.ff_net(self.ln2(x))
        logits = self.lm_head(self.ln3(x))

        if targets is None:
            loss = None
        else:
            logits = logits.reshape(B * T, -1)
            targets = targets.reshape(B * T)
            attention_mask = attention_mask.flatten() == 1

            loss = F.cross_entropy(
                logits[attention_mask, :], targets[attention_mask]
            )

        return logits, loss

    def generate(self, encoder_output, attention_mask=None, sample=False):
        B, T, C = encoder_output.shape

        # get the predictions
        logits, _ = self(encoder_output, attention_mask)  # (B, T, C)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)  # (B, C)

        if sample:
            idx_next = torch.multinomial(
                probs.view(B * self.block_size, -1),
                num_samples=1,
            )
            idx_next = idx_next.reshape(B, -1)
        else:
            idx_next = torch.argmax(probs, dim=-1)

        return idx_next  # (B, T)
