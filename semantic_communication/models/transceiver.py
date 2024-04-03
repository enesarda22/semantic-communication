from typing import Optional, List

import numpy as np
import torch
from torch import nn

from semantic_communication.models.semantic_decoder import SemanticDecoder
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.models.semantic_transformer import SemanticTransformer
from semantic_communication.utils.channel import Channel
from semantic_communication.utils.general import get_device, shift_inputs
from semantic_communication.utils.tensor_label_encoder import TensorLabelEncoder


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
        x = x / torch.norm(x, dim=2, keepdim=True)
        for l in self.layers:
            x = l(x)
        return self.linear(x)


# class SrcRelayChannelModel(nn.Module):
#     def __init__(self, n_in, n_latent, channel: Channel):
#         super().__init__()
#         self.relay_decoder = ChannelDecoder(n_latent, n_in)
#         self.channel = channel
#
#     def forward(self, src_out, d_sr):
#         ch_output = self.channel(src_out, d_sr)
#         relay_in = self.relay_decoder(ch_output)
#         return relay_in


class SrcRelayDstChannelModel(nn.Module):
    def __init__(
        self,
        n_in,
        n_latent,
        channel: Channel,
    ):
        super().__init__()
        self.relay_encoder = ChannelEncoder(n_in, n_latent)
        self.src_dst_decoder = ChannelDecoder(n_latent, n_in)
        self.relay_dst_decoder = ChannelDecoder(n_latent, n_in)
        self.channel = channel

    def forward(self, src_out, rel_x, d_rd, d_sd):
        src_dst_in = self.channel(src_out, d_sd)

        rel_out = self.relay_encoder(rel_x)
        rel_dst_in = self.channel(rel_out, d_rd)

        x_hat = torch.cat(
            [self.relay_dst_decoder(rel_dst_in), self.src_dst_decoder(src_dst_in)],
            dim=-1,
        )
        return x_hat


class SrcRelayBlock(nn.Module):
    def __init__(
        self,
        semantic_transformer: SemanticTransformer,
        src_channel_encoder: ChannelEncoder,
        relay_channel_decoder: ChannelDecoder,
        channel: Channel,
    ):
        super().__init__()
        self.semantic_encoder = semantic_transformer.semantic_encoder
        self.semantic_decoder = semantic_transformer.semantic_decoder

        self.channel = channel
        self.src_channel_encoder = src_channel_encoder
        self.relay_channel_decoder = relay_channel_decoder

    def forward(
        self,
        messages: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        d_sr: Optional[float] = None,
    ):
        x = self.semantic_encoder(
            messages=messages,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        x = self.src_channel_encoder(x)
        x = self._shift_relay_input(x)

        x = self.channel(x, d_sr)
        x = self.relay_channel_decoder(x)

        decoder_idx, targets, target_padding_mask, is_causal = shift_inputs(
            xb=input_ids,
            attention_mask=attention_mask,
            mode=self.semantic_encoder.mode,
        )

        logits, loss = self.semantic_decoder(
            idx=decoder_idx,
            encoder_output=x,
            is_causal=is_causal,
            target_padding_mask=target_padding_mask,
            targets=targets,
        )

        return logits, loss

    def _shift_relay_input(self, x):
        if self.semantic_encoder.mode == "predict":
            x = x[:, :-1, :]
        return x




# class Transceiver(nn.Module):
#     def __init__(
#         self,
#         semantic_encoder: SemanticEncoder,
#         relay_channel_block: RelayChannelBlock,
#         dst_semantic_decoder: SemanticDecoder,
#         src_relay_dst_channel_model: SrcRelayDstChannelModel,
#         label_encoder: TensorLabelEncoder,
#     ):
#         super().__init__()
#         self.semantic_encoder = semantic_encoder
#         self.src_channel_encoder = relay_channel_block.src_channel_encoder
#
#         self.src_relay_channel_model = relay_channel_block.src_relay_channel_model
#         self.relay_semantic_decoder = relay_channel_block.semantic_decoder
#         self.relay_encoder = RelayEncoder(
#             semantic_encoder=semantic_encoder,
#             label_encoder=label_encoder,
#         )
#
#         self.dst_semantic_decoder = dst_semantic_decoder
#         self.src_relay_dst_channel_model = src_relay_dst_channel_model
#
#     def forward(self, input_ids, attention_mask, targets, d_sd, d_sr, d_rd):
#         # source
#         encoder_output = self.semantic_encoder(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#         )
#         src_out = self.src_channel_encoder(encoder_output)
#         src_to_relay, src_to_dst = self._shift_src_output(src_out)
#
#         # relay
#         relay_in = self.src_relay_channel_model(src_to_relay, d_sr)
#         logits, _ = self.relay_semantic_decoder(relay_in)
#         relay_out = self.relay_encoder(logits)
#
#         # destination
#         receiver_input = self.src_relay_dst_channel_model(
#             src_to_dst, relay_out, d_rd, d_sd
#         )
#         receiver_output = self.dst_semantic_decoder(
#             encoder_output=receiver_input,
#             attention_mask=attention_mask[:, 1:],
#             targets=targets,
#         )
#         return receiver_output
#
#     def _shift_src_output(self, src_out):
#         if self.mode == "predict":
#             src_to_relay = src_out[:, :-1, :]
#             src_to_dst = src_out[:, 1:, :]
#         elif self.mode == "forward":
#             src_to_relay = src_out[:, 1:, :]
#             src_to_dst = src_out[:, 1:, :]
#         elif self.mode == "sentence":
#             src_to_relay = src_out
#             src_to_dst = src_out
#         else:
#             raise ValueError("Mode needs to be 'predict', 'forward' or 'sentence'.")
#
#         return src_to_relay, src_to_dst


class RelayEncoder:
    def __init__(
        self,
        semantic_encoder: SemanticEncoder,
        label_encoder: TensorLabelEncoder,
    ):
        super().__init__()
        self.device = get_device()
        self.semantic_encoder = semantic_encoder
        self.label_encoder = label_encoder

    def __call__(self, logits):
        B, T, _ = logits.shape
        predicted_ids = torch.argmax(logits, dim=-1)

        # append [CLS] token
        cls_padding = torch.full((B, 1), self.label_encoder.cls_id).to(self.device)
        predicted_ids = torch.cat(
            tensors=(cls_padding, predicted_ids),
            dim=1,
        )

        # transform to bert token ids
        predicted_ids = self.label_encoder.inverse_transform(predicted_ids)

        # ids are repeated to generate the embeddings sequentially
        predicted_ids = torch.repeat_interleave(predicted_ids, T, dim=0)

        # tril mask to generate the embeddings sequentially
        tril_mask = (
            torch.tril(
                torch.ones(T, T + 1, dtype=torch.long),
                diagonal=1,
            ).repeat(B, 1)
        ).to(self.device)

        out = self.semantic_encoder(
            input_ids=predicted_ids,
            attention_mask=tril_mask,
        )

        # use eye mask to select the correct embeddings sequentially
        eye_mask = (torch.eye(T).repeat(1, B) == 1).to(self.device)
        out = torch.masked_select(out[:, 1:, :].transpose(-1, 0), eye_mask)
        out = out.view(B, T, -1)

        return out
