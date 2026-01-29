import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

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
            nn.GELU(),
            nn.Linear(4 * n_embeddings, n_embeddings),  # projection
            nn.Dropout(0.1),
        )
        self.ln1 = nn.LayerNorm(n_embeddings)
        self.ln2 = nn.LayerNorm(n_embeddings)
        self.ln3 = nn.LayerNorm(n_embeddings)

        ones_tensor = torch.ones(block_size, block_size, device=self.device)
        self.register_buffer("tril", torch.tril(ones_tensor, -1).T.bool())

    def forward(
        self, x, encoder_output, source_padding_mask, enc_padding_mask, is_causal
    ):
        # norm before the layer, residual connection after the layer
        x_normed = self.ln1(x)
        seq_len = x_normed.shape[1]
        causal_mask = self.tril[:seq_len, :seq_len]
        attention_out = self.sa_heads(
            query=x_normed,
            key=x_normed,
            value=x_normed,
            key_padding_mask=source_padding_mask,
            need_weights=False,
            attn_mask=causal_mask,
            is_causal=True,
        )[0]
        x = x + attention_out

        x_normed = self.ln2(x)

        if is_causal:
            enc_seq_len = encoder_output.shape[1]
            attention_mask = self.tril[:seq_len, :enc_seq_len]
        else:
            attention_mask = None

        attention_out = self.ca_heads(
            query=x_normed,
            key=encoder_output,
            value=encoder_output,
            key_padding_mask=enc_padding_mask,
            need_weights=False,
            attn_mask=attention_mask,
            is_causal=is_causal,
        )[0]
        x = x + attention_out

        x = x + self.ff_net(self.ln3(x))
        return x, encoder_output, source_padding_mask, enc_padding_mask, is_causal


class SemanticDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_blocks,
        n_heads,
        n_embeddings,
        block_size,
        bert,
        pad_idx,
    ):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embeddings)
        self.token_embedding_table.weight = bert.embeddings.word_embeddings.weight

        self.position_embedding_table = nn.Embedding(block_size, n_embeddings)
        self.block_size = block_size

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
        self.pad_idx = pad_idx

    def forward(
        self, idx, encoder_output, is_causal, enc_padding_mask=None, targets=None
    ):
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx)  # (B,T,C)
        pos_embeddings = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T,C)
        x = token_embeddings + pos_embeddings

        source_padding_mask = idx == self.pad_idx

        x, _, _, _, _ = self.decoder_blocks(
            x, encoder_output, source_padding_mask, enc_padding_mask, is_causal
        )
        logits = self.lm_head(self.ln(x))

        if targets is None:
            loss = None
        else:
            logits = logits.reshape(B * T, -1)
            targets = targets.reshape(B * T)
            target_mask = targets != self.pad_idx

            loss = F.cross_entropy(logits[target_mask, :], targets[target_mask])

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        encoder_output,
        is_causal,
        max_length,
        enc_padding_mask=None,
        beam_width=5,
        n_generated_tokens=20,
        context_length=64,
    ):
        B = encoder_output.shape[0]
        T = n_generated_tokens

        Y = self.pad_idx * torch.ones(B, T).to(self.device).long()
        Y[:, 0] = 1

        max_context = max(1, min(context_length, self.block_size))

        def get_decoder_context(prefix, upto):
            start_idx = max(0, upto - max_context)
            return prefix[:, start_idx:upto]

        # at the start, only BOS exists, so feed it (or last K tokens)
        dec_in = get_decoder_context(Y, 1)

        next_logits, _ = self(dec_in, encoder_output, is_causal, enc_padding_mask)
        next_logits = next_logits[:, -1, :]  # last position predicts next token
        vocab_size = next_logits.shape[-1]

        probabilities, next_chars = F.log_softmax(next_logits, dim=-1).topk(
            k=beam_width, dim=-1
        )

        Y = Y.repeat_interleave(beam_width, dim=0)
        encoder_output = encoder_output.repeat_interleave(beam_width, dim=0)
        if enc_padding_mask is not None:
            enc_padding_mask = enc_padding_mask.repeat_interleave(beam_width, dim=0)

        Y[:, 1] = next_chars.flatten()

        for i in range(1, T - 1):
            # We are about to predict Y[:, i+1].
            # The known prefix is Y[:, :i+1]. Use only last K tokens of that prefix.
            dec_in = get_decoder_context(Y, i + 1)  # shape: [B*beam, <=K]

            next_logits, _ = self(dec_in, encoder_output, is_causal, enc_padding_mask)

            # take logits at the last position of the truncated input
            next_logits = next_logits[:, -1, :]  # shape: [B*beam, vocab]
            next_probabilities = F.log_softmax(next_logits, dim=-1)

            next_probabilities = next_probabilities.reshape(
                (-1, beam_width, next_probabilities.shape[-1])
            )
            probabilities = probabilities.unsqueeze(-1) + next_probabilities
            probabilities = probabilities.flatten(start_dim=1)
            probabilities, idx = probabilities.topk(k=beam_width, axis=-1)
            next_chars = torch.remainder(idx, vocab_size).flatten().unsqueeze(-1)

            best_candidates = (idx / vocab_size).long()
            best_candidates += (
                    torch.arange(Y.shape[0] // beam_width,
                                 device=self.device).unsqueeze(-1)
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

        return Y, probabilities[torch.arange(B), best_indices]

    @torch.no_grad()
    def generate_greedy(
        self,
        encoder_output,
        is_causal,
        max_length,
        enc_padding_mask=None,
        n_generated_tokens=20,
    ):
        B = encoder_output.shape[0]
        T = n_generated_tokens

        Y = self.pad_idx * torch.ones(B, T).to(self.device).long()
        Y[:, 0] = 1
        next_logits, _ = self(
            Y[:, :max_length], encoder_output, is_causal, enc_padding_mask
        )
        next_logits = next_logits[:, 0, :]

        next_tokens = torch.argmax(next_logits, dim=-1)
        Y[:, 1] = next_tokens.flatten()

        for i in range(1, T - 1):
            start_idx = max(i - max_length, 0)
            end_idx = start_idx + max_length
            next_logits, _ = self(
                Y[:, -start_idx:end_idx], encoder_output, is_causal, enc_padding_mask
            )
            next_logits = next_logits[:, i, :]

            next_tokens = torch.argmax(next_logits, dim=-1)
            Y[:, i + 1] = next_tokens.flatten()

            if torch.all(torch.any(Y == 2, dim=1)):
                break

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
