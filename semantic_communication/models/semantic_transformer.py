from typing import Optional, List

import torch
from torch import nn

from semantic_communication.models.semantic_decoder import SemanticDecoder
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.utils.general import shift_inputs


class SemanticTransformer(nn.Module):
    def __init__(
        self,
        semantic_encoder: SemanticEncoder,
        semantic_decoder: SemanticDecoder,
    ):
        super().__init__()
        self.semantic_encoder = semantic_encoder
        self.semantic_decoder = semantic_decoder
        self.mode = semantic_encoder.mode

    def forward(
        self,
        messages: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        snr_db: Optional[float] = None,
    ):
        encoder_output = self.semantic_encoder(
            messages=messages,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        encoder_output = self._add_noise(encoder_output, snr_db)

        decoder_idx, targets, enc_padding_mask, is_causal = shift_inputs(
            xb=input_ids,
            attention_mask=attention_mask,
            mode=self.mode,
        )

        logits, loss = self.semantic_decoder(
            idx=decoder_idx,
            encoder_output=encoder_output,
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
    ):
        self.eval()
        with torch.no_grad():
            encoder_output = self.semantic_encoder(
                messages=messages,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            encoder_output = self._add_noise(encoder_output, snr_db)

            return self.semantic_decoder.generate(
                encoder_output=encoder_output,
                beam_width=beam_width,
                max_length=max_length,
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
