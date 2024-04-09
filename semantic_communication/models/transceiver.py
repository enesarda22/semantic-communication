from typing import Optional, List

import numpy as np
import torch
from torch import nn

from semantic_communication.models.semantic_decoder import SemanticDecoder
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.models.semantic_transformer import SemanticTransformer
from semantic_communication.utils.channel import Channel
from semantic_communication.utils.general import get_device, shift_inputs


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


# class SrcRelayDstChannelModel(nn.Module):
#     def __init__(
#         self,
#         n_in,
#         n_latent,
#         channel: Channel,
#     ):
#         super().__init__()
#         self.relay_encoder = ChannelEncoder(n_in, n_latent)
#         self.src_dst_decoder = ChannelDecoder(n_latent, n_in)
#         self.relay_dst_decoder = ChannelDecoder(n_latent, n_in)
#         self.channel = channel
#
#     def forward(self, src_out, rel_x, d_rd, d_sd):
#         src_dst_in = self.channel(src_out, d_sd)
#
#         rel_out = self.relay_encoder(rel_x)
#         rel_dst_in = self.channel(rel_out, d_rd)
#
#         x_hat = torch.cat(
#             [self.relay_dst_decoder(rel_dst_in), self.src_dst_decoder(src_dst_in)],
#             dim=-1,
#         )
#         return x_hat


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


class Transceiver(nn.Module):
    def __init__(
        self,
        src_relay_block: SrcRelayBlock,
        relay_semantic_encoder: SemanticEncoder,
        relay_channel_encoder: ChannelEncoder,
        dst_channel_decoder: ChannelDecoder,
        dst_semantic_decoder: SemanticDecoder,
        channel: Channel,
        max_length: int,
    ):
        super().__init__()
        # source
        self.src_semantic_encoder = src_relay_block.semantic_encoder
        self.src_channel_encoder = src_relay_block.src_channel_encoder

        # relay
        self.relay_channel_decoder = src_relay_block.relay_channel_decoder
        self.relay_semantic_decoder = src_relay_block.semantic_decoder
        self.relay_semantic_encoder = relay_semantic_encoder
        self.relay_channel_encoder = relay_channel_encoder

        # destination
        self.dst_channel_decoder = dst_channel_decoder
        self.dst_semantic_decoder = dst_semantic_decoder

        self.channel = channel
        self.max_length = max_length
        self.device = get_device()

    def forward(
        self,
        messages: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        d_sd: Optional[float] = None,
        d_sr: Optional[float] = None,
    ):
        # source
        x_src = self.src_semantic_encoder(
            messages=messages,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        x_src = self.src_channel_encoder(x_src)
        x_src_to_relay, x_src_to_dst = self._shift_src_output(x_src)

        # relay
        x_relay = self.channel(x_src_to_relay, d_sr)
        x_relay = self.relay_channel_decoder(x_relay)
        x_relay, _ = self.relay_semantic_decoder.generate(
            encoder_output=x_relay,
            is_causal=self.relay_semantic_encoder.mode != "sentence",
            max_length=self.max_length,  # TODO: fix +1 discrepancy
            n_generated_tokens=self.max_length + 1,
        )  # TODO: transceiver sees all the embeddings?

        relay_attention_mask = torch.ones(
            *x_relay.shape, dtype=torch.long, device=self.device
        )
        for i in range(x_relay.shape[0]):
            k = torch.argmax((x_relay[i] == 2).long()).item()
            if k == 0:
                continue
            relay_attention_mask[i, k + 1 :] = 0

        x_relay = self.relay_semantic_encoder(
            input_ids=x_relay,
            attention_mask=relay_attention_mask,
        )
        x_relay = self.relay_channel_encoder(x_relay)

        # destination
        x_dst1 = self.channel(x_relay, d_sd - d_sr)
        x_dst2 = self.channel(x_src_to_dst, d_sd)
        x_dst = torch.cat((x_dst1, x_dst2), dim=-1)
        x_dst = self.dst_channel_decoder(x_dst)

        decoder_idx, targets, enc_padding_mask, is_causal = shift_inputs(
            xb=input_ids,
            attention_mask=attention_mask,
            mode=self.relay_semantic_encoder.mode,
        )

        logits, loss = self.dst_semantic_decoder(
            idx=decoder_idx,
            encoder_output=x_dst,
            is_causal=is_causal,
            enc_padding_mask=enc_padding_mask,
            targets=targets,
        )

        return logits, loss

    def _shift_src_output(self, src_out):
        if self.relay_semantic_encoder.mode == "predict":
            src_to_relay = src_out[:, :-1, :]
            src_to_dst = src_out[:, 1:, :]
        else:
            src_to_relay = src_out
            src_to_dst = src_out

        return src_to_relay, src_to_dst


# class RelayEncoder:
#     def __init__(
#         self,
#         semantic_encoder: SemanticEncoder,
#         label_encoder: TensorLabelEncoder,
#     ):
#         super().__init__()
#         self.device = get_device()
#         self.semantic_encoder = semantic_encoder
#         self.label_encoder = label_encoder
#
#     def __call__(self, logits):
#         B, T, _ = logits.shape
#         predicted_ids = torch.argmax(logits, dim=-1)
#
#         # append [CLS] token
#         cls_padding = torch.full((B, 1), self.label_encoder.cls_id).to(self.device)
#         predicted_ids = torch.cat(
#             tensors=(cls_padding, predicted_ids),
#             dim=1,
#         )
#
#         # transform to bert token ids
#         predicted_ids = self.label_encoder.inverse_transform(predicted_ids)
#
#         # ids are repeated to generate the embeddings sequentially
#         predicted_ids = torch.repeat_interleave(predicted_ids, T, dim=0)
#
#         # tril mask to generate the embeddings sequentially
#         tril_mask = (
#             torch.tril(
#                 torch.ones(T, T + 1, dtype=torch.long),
#                 diagonal=1,
#             ).repeat(B, 1)
#         ).to(self.device)
#
#         out = self.semantic_encoder(
#             input_ids=predicted_ids,
#             attention_mask=tril_mask,
#         )
#
#         # use eye mask to select the correct embeddings sequentially
#         eye_mask = (torch.eye(T).repeat(1, B) == 1).to(self.device)
#         out = torch.masked_select(out[:, 1:, :].transpose(-1, 0), eye_mask)
#         out = out.view(B, T, -1)
#
#         return out
