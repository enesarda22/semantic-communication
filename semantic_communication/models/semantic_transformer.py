from typing import Optional, List

import torch
from torch import nn

from semantic_communication.models.semantic_decoder import SemanticDecoder
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.utils.general import (
    shift_inputs,
    get_device,
)


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
    ):
        encoder_output = self.semantic_encoder(
            messages=messages,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        decoder_idx, decoder_attention_mask, targets = shift_inputs(
            xb=input_ids,
            attention_mask=attention_mask,
            mode=self.mode,
        )

        logits, loss = self.semantic_decoder(
            decoder_idx, encoder_output, decoder_attention_mask, targets
        )

        return logits, loss

    def generate(
        self,
        messages: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
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

            return self.semantic_decoder.generate(
                encoder_output=encoder_output,
                beam_width=beam_width,
                max_length=max_length,
            )
