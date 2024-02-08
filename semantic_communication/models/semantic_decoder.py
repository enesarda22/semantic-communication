import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

from semantic_communication.utils.general import get_device


class MultiInputSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class DecoderBlock(nn.Module):
    def __init__(self, n_heads, n_embeddings, block_size):
        super().__init__()
        self.device = get_device()
        self.sa_heads = nn.MultiheadAttention(
            embed_dim=n_embeddings,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.ca_heads = nn.MultiheadAttention(
            embed_dim=n_embeddings,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True,
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

        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size, device=self.device))
        )

    def forward(self, x, encoder_output, attention_mask):
        # norm before the layer, residual connection after the layer
        x_normed = self.ln1(x)
        attention_out = self.sa_heads(
            query=x_normed,
            key=x_normed,
            value=x_normed,
            key_padding_mask=(attention_mask == 0),
            need_weights=False,
            attn_mask=(self.tril == 0),
            is_causal=True,
        )[0]
        x = x + attention_out

        x_normed = self.ln2(x)

        # prepare masks for cross attention heads
        if encoder_output.shape[1] == self.tril.shape[1]:
            attn_mask = self.tril == 0
            key_padding_mask = attention_mask == 0
            is_causal = True
        else:
            attn_mask = torch.zeros(
                (self.tril.shape[0], encoder_output.shape[1]),
                dtype=torch.bool,
                device=self.device,
            )
            key_padding_mask = torch.zeros(
                (encoder_output.shape[:2]),
                dtype=torch.bool,
                device=self.device,
            )
            is_causal = False

        attention_out = self.ca_heads(
            query=x_normed,
            key=encoder_output,
            value=encoder_output,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )[0]
        x = x + attention_out

        x = x + self.ff_net(self.ln3(x))
        return x, encoder_output, attention_mask


class SemanticDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_blocks,
        n_heads,
        n_embeddings,
        block_size,
        bert,
    ):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embeddings)
        self.token_embedding_table.weight = bert.embeddings.word_embeddings.weight

        self.position_embedding_table = nn.Embedding(block_size, n_embeddings)

        self.decoder_blocks = MultiInputSequential(
            *[
                DecoderBlock(
                    n_heads=n_heads,
                    n_embeddings=n_embeddings,
                    block_size=block_size,
                )
                for _ in range(n_blocks)
            ]
        )
        self.ln = nn.LayerNorm(n_embeddings)
        self.lm_head = nn.Linear(n_embeddings, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight

        self.device = get_device()

    def forward(self, idx, encoder_output, attention_mask=None, targets=None):
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx)  # (B,T,C)
        pos_embeddings = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T,C)
        x = token_embeddings + pos_embeddings

        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.long).to(self.device)

        x, _, _ = self.decoder_blocks(x, encoder_output, attention_mask)
        logits = self.lm_head(self.ln(x))

        if targets is None:
            loss = None
        else:
            logits = logits.reshape(B * T, -1)
            targets = targets.reshape(B * T)
            attention_mask = attention_mask.flatten() == 1

            loss = F.cross_entropy(logits[attention_mask, :], targets[attention_mask])

        return logits, loss

    def generate(
        self,
        encoder_output,
        beam_width=5,
        max_length=20,
    ):
        B = encoder_output.shape[0]
        T = max_length

        with torch.no_grad():
            Y = torch.zeros(B, T).to(self.device).long()
            Y[:, 0] = 1

            attn_mask = torch.zeros(B, T).to(self.device).long()
            attn_mask[:, 0] = 1

            next_logits, _ = self(Y, encoder_output, attn_mask)
            next_logits = next_logits[:, 0, :]
            vocab_size = next_logits.shape[-1]

            probabilities, next_chars = F.log_softmax(next_logits, dim=-1).topk(
                k=beam_width, dim=-1
            )

            Y = Y.repeat((beam_width, 1))
            Y[:, 1] = next_chars.flatten()

            for i in tqdm(range(1, max_length - 1)):
                attn_mask[:, i] = 1

                dataset = TensorDataset(
                    Y[:, -max_length:],
                    encoder_output.repeat((beam_width, 1, 1, 1))
                    .transpose(0, 1)
                    .flatten(end_dim=1),
                    attn_mask.repeat((beam_width, 1)),
                )
                dl = DataLoader(dataset, batch_size=32)
                next_probabilities = []

                for x, e, mask in tqdm(dl):
                    next_logits, _ = self(x, e, mask)
                    next_logits = next_logits[:, i, :]
                    next_probabilities.append(F.log_softmax(next_logits, dim=-1))

                next_probabilities = torch.cat(next_probabilities, axis=0)
                next_probabilities = next_probabilities.reshape(
                    (-1, beam_width, next_probabilities.shape[-1])
                )
                probabilities = probabilities.unsqueeze(-1) + next_probabilities
                probabilities = probabilities.flatten(start_dim=1)
                probabilities, idx = probabilities.topk(k=beam_width, axis=-1)
                next_chars = torch.remainder(idx, vocab_size).flatten().unsqueeze(-1)

                best_candidates = (idx / vocab_size).long()
                best_candidates += (
                    torch.arange(
                        Y.shape[0] // beam_width, device=self.device
                    ).unsqueeze(-1)
                    * beam_width
                )

                Y = Y[best_candidates].flatten(end_dim=-2)
                Y[:, i + 1] = next_chars.flatten()

                if torch.all(torch.any(Y == 2, dim=1)):
                    break

            best_indices = torch.argmax(probabilities, dim=1)
            Y = torch.gather(
                Y.reshape(-1, beam_width, Y.shape[-1]),
                1,
                best_indices.reshape(-1, 1, 1).repeat((1, 1, Y.shape[-1])),
            ).squeeze(1)

            return Y

    def generate_next(
        self,
        idx,
        encoder_output,
        attention_mask=None,
        sample=False,
    ):
        B, T, C = encoder_output.shape

        # get the predictions
        logits, _ = self(idx, encoder_output, attention_mask)  # (B, T, C)
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
