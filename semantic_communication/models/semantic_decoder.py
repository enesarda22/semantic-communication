import torch
import torch.nn as nn
from torch.nn import functional as F

from semantic_communication.models.multi_head_attention import MultiHeadAttention


class SemanticDecoder(nn.Module):
    def __init__(self, vocab_size, n_heads, n_embeddings, block_size, device):
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
        self.device = device

    def forward(self, encoder_output, targets=None):
        # residual connection after the layer, norm before the layer
        x = encoder_output + self.sa_heads(self.ln1(encoder_output))
        x = x + self.ff_net(self.ln2(x))
        logits = self.lm_head(self.ln3(x))

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.reshape(B * T, C)
            targets = targets.reshape(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, encoder_output, sample=False):
        B, T, C = encoder_output.shape

        padded_encoder_output = torch.ones((B, self.block_size, C))
        padded_encoder_output[:, :T, :] = encoder_output

        # get the predictions
        logits, _ = self(padded_encoder_output)  # (B, T, C)
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

        return idx_next[:, :T]  # (B, T)
