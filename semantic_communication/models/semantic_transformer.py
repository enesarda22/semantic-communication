import torch
from torch import nn

from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.utils.general import (
    shift_inputs,
    get_device,
)


class SemanticTransformer(nn.Module):
    def __init__(self, semantic_encoder, semantic_decoder, mode):
        super().__init__()
        self.semantic_encoder = semantic_encoder
        self.semantic_decoder = semantic_decoder
        self.mode = mode
        self.device = get_device()

    def forward(
        self,
        encoder_idx,
        encoder_attention_mask,
    ):
        encoder_output = self.get_encoder_output(
            encoder_idx=encoder_idx,
            encoder_attention_mask=encoder_attention_mask,
        )

        decoder_idx, decoder_attention_mask, targets = shift_inputs(
            xb=encoder_idx,
            attention_mask=encoder_attention_mask,
            mode=self.mode,
        )

        logits, loss = self.semantic_decoder(
            decoder_idx, encoder_output, decoder_attention_mask, targets
        )

        return logits, loss

    def get_encoder_output(self, encoder_idx, encoder_attention_mask):
        encoder_lhs = self.semantic_encoder(
            input_ids=encoder_idx,
            attention_mask=encoder_attention_mask,
        )["last_hidden_state"]

        encoder_output = SemanticEncoder.mean_pooling(
            bert_lhs=encoder_lhs,
            attention_mask=encoder_attention_mask,
        )

        encoder_output = torch.cat(
            tensors=(encoder_output.unsqueeze(1), encoder_lhs[:, 1:, :]),
            dim=1,
        )

        if self.mode == "predict":
            encoder_output = encoder_output[:, :-1, :]
        elif self.mode == "forward":
            encoder_output = encoder_output[:, 1:, :]
        elif self.mode == "sentence":
            encoder_output = encoder_output[:, [0], :]
        else:
            raise ValueError("Mode needs to be 'predict', 'forward' or 'sentence'.")

        return encoder_output

    def generate(
        self,
        encoder_idx,
        encoder_attention_mask,
        beam_width=5,
        max_length=20,
    ):
        self.eval()
        with torch.no_grad():
            encoder_output = self.get_encoder_output(
                encoder_idx=encoder_idx,
                encoder_attention_mask=encoder_attention_mask,
            )

            return self.semantic_decoder.generate(
                encoder_output=encoder_output,
                beam_width=beam_width,
                max_length=max_length,
            )
