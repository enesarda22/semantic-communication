from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math

from semantic_communication.models.semantic_decoder import SemanticDecoder
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.utils.channel import Channel
from semantic_communication.utils.general import shift_inputs, get_device, pad_cls


class ChannelEncComp(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ChannelEncComp, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.ln(x)
        out = self.prelu(x)
        return out


class ChannelEncoder(nn.Module):
    def __init__(self, nin, nout):
        super(ChannelEncoder, self).__init__()
        up_dim = int(np.floor(np.log2(nin) / 2))
        low_dim = int(np.ceil(np.log2(nout) / 2))

        log_val = math.log(nin, 4)
        if not int(log_val) == log_val:
            dims = [nin]
        else:
            dims = []

        for i in range(up_dim - low_dim + 1):
            dims.append(np.power(4, up_dim - i))

        self.layers = nn.ModuleList(
            [ChannelEncComp(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        self.linear = nn.Linear(dims[-1], nout)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return self.linear(x)


class ChannelDecoder(nn.Module):
    def __init__(self, nin, nout):
        super(ChannelDecoder, self).__init__()
        up_dim = int(np.floor(np.log2(nout) / 2))
        low_dim = int(np.ceil(np.log2(nin) / 2))
        dims = [nin]
        for i in range(up_dim - low_dim + 1):
            dims.append(np.power(4, low_dim + i))

        self.layers = nn.ModuleList(
            [ChannelEncComp(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        self.linear = nn.Linear(dims[-1], nout)

    def forward(self, x):
        # x = x / torch.norm(x, dim=2, keepdim=True)  # TODO: do not normalize
        for l in self.layers:
            x = l(x)
        return self.linear(x)


class SemanticTransformer(nn.Module):
    def __init__(
        self,
        semantic_encoder: SemanticEncoder,
        semantic_decoder: SemanticDecoder,
        channel_encoder: ChannelEncoder,
        channel_decoder: ChannelDecoder,
        channel: Optional[Channel] = None,
    ):
        super().__init__()
        self.semantic_encoder = semantic_encoder
        self.semantic_decoder = semantic_decoder
        self.channel_encoder = channel_encoder
        self.channel_decoder = channel_decoder
        self.max_length = semantic_encoder.max_length - 1

        self.channel = channel
        self.mode = semantic_encoder.mode
        self.device = get_device()

    def forward(
        self,
        messages: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        snr_db: Optional[float] = None,
        d: Optional[float] = None,
    ):
        x = self.semantic_encoder(
            messages=messages,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        x, _ = self.shift_src_output(x, mode=self.mode)

        x = self.channel_encoder(x)

        if self.channel is None:
            # signal power constraint
            last_dim = x.shape[-1]
            x = ((last_dim / 2) ** 0.5) * F.normalize(x, dim=-1, p=2)

            x = self._add_noise(x, snr_db)
        else:
            x = self.channel(x=x, d=d)

        x = self.channel_decoder(x)

        if self.semantic_encoder.mode == "next_sentence":
            input_ids = pad_cls(input_ids[:, -self.max_length :])

        decoder_idx, targets, enc_padding_mask, is_causal = shift_inputs(
            xb=input_ids,
            attention_mask=attention_mask,
            mode=self.mode,
        )

        logits, loss = self.semantic_decoder(
            idx=decoder_idx,
            encoder_output=x,
            is_causal=is_causal,
            enc_padding_mask=enc_padding_mask,
            targets=targets,
        )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        messages: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        snr_db: Optional[float] = None,
        d: Optional[float] = None,
        beam_width=5,
    ):
        x = self.semantic_encoder(
            messages=messages,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        x, _ = self.shift_src_output(x, mode=self.mode)

        x = self.channel_encoder(x)

        if self.channel is None:
            # signal power constraint
            gain = torch.sqrt(0.5 / torch.var(x, dim=-1))
            x = x * gain[:, :, None]

            x = self._add_noise(x, snr_db)
        else:
            x = self.channel(x=x, d=d)

        x = self.channel_decoder(x)

        if self.mode == "sentence" or self.mode == "next_sentence":
            return self.generate_sentence(x=x, beam_width=beam_width, greedy=False)[0]
        else:
            return self.generate_token(
                x=x, beam_width=beam_width, attention_mask=attention_mask, greedy=True
            )

    def generate_sentence(self, x, beam_width, greedy):
        B, R, _ = x.shape
        x = torch.repeat_interleave(input=x, repeats=R, dim=0)

        x_padding_mask = torch.tril(torch.ones(R, R, device=self.device), -1).T.bool()
        x_padding_mask = x_padding_mask.repeat(B, 1)

        if greedy:
            return self.semantic_decoder.generate_greedy(
                encoder_output=x,
                is_causal=False,
                max_length=self.max_length,
                enc_padding_mask=x_padding_mask,
                n_generated_tokens=self.max_length + 1,
            )
        else:
            return self.semantic_decoder.generate(
                encoder_output=x,
                is_causal=False,
                max_length=self.max_length,
                enc_padding_mask=x_padding_mask,
                beam_width=beam_width,
                n_generated_tokens=self.max_length + 1,
            )

    def generate_token(self, x, beam_width, attention_mask, greedy):
        x_padding_mask = attention_mask[:, 1:] == 0
        is_causal = True

        if greedy:
            return self.semantic_decoder.generate_greedy(
                encoder_output=x,
                is_causal=is_causal,
                max_length=self.max_length,
                enc_padding_mask=x_padding_mask,
                n_generated_tokens=self.max_length + 1,
            )
        else:
            return self.semantic_decoder.generate(
                encoder_output=x,
                is_causal=is_causal,
                max_length=self.max_length,
                enc_padding_mask=x_padding_mask,
                beam_width=beam_width,
                n_generated_tokens=self.max_length + 1,
            )

    @staticmethod
    def _add_noise(signal, snr_db):
        if snr_db is not None:
            signal_pow = torch.mean(torch.pow(signal, 2), dim=-1, keepdim=True)
            noise_pow = signal_pow / (10 ** (snr_db / 10))

            noise = torch.sqrt(noise_pow) * torch.randn(
                size=signal.shape, device=signal.device
            )
            return signal + noise

        else:
            return signal

    @staticmethod
    def shift_src_output(src_out, mode):
        if mode == "predict":
            src_to_relay = src_out[:, :-1, :]
            src_to_dst = src_out[:, 1:, :]
        else:
            src_to_relay = src_out
            src_to_dst = src_out

        return src_to_relay, src_to_dst
