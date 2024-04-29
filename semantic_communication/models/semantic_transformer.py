from typing import Optional, List

import numpy as np
import torch
from torch import nn

from semantic_communication.models.semantic_decoder import SemanticDecoder
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.utils.general import shift_inputs


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

        dims = [nin]
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
    ):
        super().__init__()
        self.semantic_encoder = semantic_encoder
        self.semantic_decoder = semantic_decoder
        self.channel_encoder = channel_encoder
        self.channel_decoder = channel_decoder
        self.mode = semantic_encoder.mode

    def forward(
        self,
        messages: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        snr_db: Optional[float] = None,
    ):
        x = self.semantic_encoder(
            messages=messages,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        x = self.channel_encoder(x)

        # signal power constraint
        last_dim = int(x.shape[-1] / 2)
        x = torch.complex(*torch.split(x, last_dim, dim=-1))
        x = x / torch.abs(x)
        x = torch.cat((x.real, x.imag), dim=-1)

        x = self._add_noise(x, snr_db)
        x = self.channel_decoder(x)

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

    def generate(
        self,
        messages: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        snr_db: Optional[float] = None,
        beam_width=5,
        max_length=20,
        n_generated_tokens=20,
    ):
        with torch.no_grad():
            encoder_output = self.semantic_encoder(
                messages=messages,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            encoder_output = self._add_noise(encoder_output, snr_db)

            return self.semantic_decoder.generate(
                encoder_output=encoder_output,
                is_causal=False,
                max_length=max_length,
                enc_padding_mask=None,
                beam_width=beam_width,
                n_generated_tokens=n_generated_tokens,
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
