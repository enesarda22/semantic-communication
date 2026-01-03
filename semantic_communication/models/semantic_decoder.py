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
    def __init__(self, n_heads, n_embeddings, block_size, state_memory_len=-1):
        super().__init__()
        self.device = get_device()
        self.state_memory_len = state_memory_len
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
        attention_out = self.sa_heads(
            query=x_normed,
            key=x_normed,
            value=x_normed,
            key_padding_mask=source_padding_mask,
            need_weights=False,
            attn_mask=self.tril,
            is_causal=True,
        )[0]
        x = x + attention_out

        x_normed = self.ln2(x)

        if is_causal:
            base = self._cross_attn_mask(
                Tq=x_normed.size(1),
                Tk=encoder_output.size(1),
                device=x_normed.device,
            )  # (Tq, Tk) bool; True = masked

            # If we're doing a finite memory window, we need per-sample safety w.r.t. enc_padding_mask
            if (
                self.state_memory_len is not None
                and self.state_memory_len >= 0
                and enc_padding_mask is not None
            ):
                B, Tq, Tk = (
                    encoder_output.size(0),
                    x_normed.size(1),
                    encoder_output.size(1),
                )
                H = self.ca_heads.num_heads

                # (B, Tq, Tk)
                attn_mask = base.unsqueeze(0).expand(B, Tq, Tk).clone()

                # For padded *queries*, don't mask anything (loss ignores them anyway)
                if source_padding_mask is not None:
                    attn_mask[source_padding_mask] = (
                        False  # unmask all keys for those query rows
                    )

                valid_k = ~enc_padding_mask  # (B, Tk)

                # Rows that would have zero valid keys after combining with key padding
                allowed = (~attn_mask) & valid_k.unsqueeze(1)  # (B, Tq, Tk)
                row_bad = ~allowed.any(dim=-1)  # (B, Tq)

                if row_bad.any():
                    # last valid key index per sample (padding is at the end, so this works)
                    last_valid = valid_k.long().sum(dim=1) - 1  # (B,)
                    b_idx, q_idx = row_bad.nonzero(as_tuple=True)
                    k_idx = last_valid[b_idx].clamp(min=0)
                    attn_mask[b_idx, q_idx, k_idx] = (
                        False  # ensure at least 1 key is unmasked
                    )

                # MultiheadAttention wants (B*H, Tq, Tk) for per-sample masks
                attention_mask = attn_mask.repeat_interleave(H, dim=0)

            else:
                # either full memory (-1) or no enc_padding_mask => 2D mask is fine
                attention_mask = base
        else:
            attention_mask = None

        attention_out = self.ca_heads(
            query=x_normed,
            key=encoder_output,
            value=encoder_output,
            key_padding_mask=enc_padding_mask,
            need_weights=False,
            attn_mask=attention_mask,
            is_causal=False,
        )[0]
        x = x + attention_out

        x = x + self.ff_net(self.ln3(x))
        return x, encoder_output, source_padding_mask, enc_padding_mask, is_causal

    def _cross_attn_mask(self, Tq: int, Tk: int, device):
        # bool mask: True means "masked out"
        if self.state_memory_len is None or self.state_memory_len < 0:
            return self.tril[:Tq, :Tk]

        window = max(1, int(self.state_memory_len))
        window = min(window, Tk)

        q = torch.arange(Tq, device=device)[:, None]  # (Tq,1)
        k = torch.arange(Tk, device=device)[None, :]  # (1,Tk)

        mask_future = k > q
        mask_too_old = k < (q - (window - 1))
        return mask_future | mask_too_old


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
        state_memory_len=-1,
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
                    state_memory_len=state_memory_len,
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
    ):
        B = encoder_output.shape[0]
        T = n_generated_tokens

        Y = self.pad_idx * torch.ones(B, T).to(self.device).long()
        Y[:, 0] = 1

        next_logits, _ = self(
            Y[:, :max_length], encoder_output, is_causal, enc_padding_mask
        )
        next_logits = next_logits[:, 0, :]
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
            start_idx = max(i - max_length, 0)
            end_idx = start_idx + max_length

            if enc_padding_mask is None:
                dataset = TensorDataset(Y[:, -start_idx:end_idx], encoder_output)
            else:
                dataset = TensorDataset(
                    Y[:, -start_idx:end_idx], encoder_output, enc_padding_mask
                )

            dl = DataLoader(dataset, batch_size=B)
            next_probabilities = []

            for x in dl:
                if enc_padding_mask is None:
                    next_logits, _ = self(x[0], x[1], is_causal, None)
                else:
                    next_logits, _ = self(x[0], x[1], is_causal, x[2])
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
                torch.arange(Y.shape[0] // beam_width, device=self.device).unsqueeze(-1)
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
