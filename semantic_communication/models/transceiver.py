import copy
from typing import Optional, List

import torch
from torch import nn

from semantic_communication.models.semantic_decoder import SemanticDecoder
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.models.semantic_transformer import (
    SemanticTransformer,
    ChannelEncoder,
    ChannelDecoder,
)
from semantic_communication.utils.channel import Channel
from semantic_communication.utils.general import get_device, shift_inputs


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


# class SrcRelayBlock(nn.Module):
#     def __init__(
#         self,
#         semantic_transformer: SemanticTransformer,
#         channel: Channel,
#     ):
#         super().__init__()
#         self.semantic_encoder = semantic_transformer.semantic_encoder
#         self.semantic_decoder = semantic_transformer.semantic_decoder
#
#         self.channel = channel
#         self.src_channel_encoder = semantic_transformer.channel_encoder
#         self.relay_channel_decoder = semantic_transformer.channel_decoder
#
#         self.device = get_device()
#
#     def forward(
#         self,
#         messages: Optional[List[str]] = None,
#         input_ids: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         d_sr: Optional[float] = None,
#     ):
#         x = self.semantic_encoder(
#             messages=messages,
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#         )
#         x = self.src_channel_encoder(x)
#         x = self._shift_relay_input(x)
#
#         x = self.channel(x, d_sr)
#         x = self.relay_channel_decoder(x)
#
#         B, R, C = x.shape
#         x = torch.repeat_interleave(input=x, repeats=R, dim=0)
#
#         enc_padding_mask = torch.tril(torch.ones(R, R, device=self.device), -1).T.bool()
#         enc_padding_mask = enc_padding_mask.repeat(B, 1)
#
#         decoder_idx, targets, _, is_causal = shift_inputs(
#             xb=input_ids,
#             attention_mask=attention_mask,
#             mode=self.semantic_encoder.mode,
#             rate=R,
#         )
#
#         decoder_idx = torch.repeat_interleave(input=decoder_idx, repeats=R, dim=0)
#         targets = torch.repeat_interleave(input=targets, repeats=R, dim=0)
#
#         logits, loss = self.semantic_decoder(
#             idx=decoder_idx,
#             encoder_output=x,
#             is_causal=is_causal,
#             enc_padding_mask=enc_padding_mask,
#             targets=targets,
#         )
#
#         return logits, loss
#
#     @torch.no_grad()
#     def generate(
#         self,
#         messages: Optional[List[str]] = None,
#         input_ids: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         d_sr: Optional[float] = None,
#     ):
#         x = self.semantic_encoder(
#             messages=messages,
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#         )
#         x = self.src_channel_encoder(x)
#         x = self._shift_relay_input(x)
#
#         x = self.channel(x, d_sr)
#         x = self.relay_channel_decoder(x)
#
#         B, R, _ = x.shape
#         x = torch.repeat_interleave(input=x, repeats=R, dim=0)
#
#         x_padding_mask = torch.tril(torch.ones(R, R, device=self.device), -1).T.bool()
#         x_padding_mask = x_padding_mask.repeat(B, 1)
#
#         return self.semantic_decoder.generate(
#             encoder_output=x,
#             is_causal=False,
#             max_length=self.semantic_encoder.max_length - 1,
#             enc_padding_mask=x_padding_mask,
#             n_generated_tokens=self.semantic_encoder.max_length,
#         )
#
#     def _shift_relay_input(self, x):
#         if self.semantic_encoder.mode == "predict":
#             x = x[:, :-1, :]
#         return x


class Transceiver(nn.Module):
    def __init__(
        self,
        src_relay_transformer: SemanticTransformer,
        relay_semantic_encoder: SemanticEncoder,
        relay_channel_encoder: ChannelEncoder,
        dst_channel_decoder: ChannelDecoder,
        dst_semantic_decoder: SemanticDecoder,
        channel: Channel,
        max_length: int,
    ):
        super().__init__()
        # source
        self.src_semantic_encoder = src_relay_transformer.semantic_encoder
        self.src_channel_encoder = src_relay_transformer.channel_encoder

        # freeze source params
        self._freeze(self.src_semantic_encoder)
        self._freeze(self.src_channel_encoder)

        # relay
        self.relay_channel_decoder = src_relay_transformer.channel_decoder
        self.relay_semantic_decoder = src_relay_transformer.semantic_decoder

        # freeze relay decoding params
        self._freeze(self.relay_channel_decoder)
        self._freeze(self.relay_semantic_decoder)

        self.relay_semantic_encoder = relay_semantic_encoder
        self.relay_channel_encoder = relay_channel_encoder

        # destination
        self.dst_channel_decoder = dst_channel_decoder
        self.dst_semantic_decoder = dst_semantic_decoder

        self.channel = channel
        self.max_length = max_length
        self.device = get_device()

    @staticmethod
    def _freeze(m):
        for p in m.parameters():
            p.requires_grad = False

    def forward(
        self,
        messages: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        d_sd: Optional[float] = None,
        d_sr: Optional[float] = None,
    ):
        # source
        x_src_to_dst, x_src_to_relay = self._source_forward(
            attention_mask=attention_mask,
            input_ids=input_ids,
            messages=messages,
        )

        # relay
        x_relay = self.channel(x_src_to_relay, d_sr)
        x_relay = self._relay_forward(
            x_relay=x_relay,
            attention_mask=attention_mask,
        )

        # destination
        x_dst1 = self.channel(x_relay, d_sd - d_sr)
        x_dst2 = self.channel(x_src_to_dst, d_sd)
        x_dst = torch.cat((x_dst1, x_dst2), dim=-1)

        logits, loss = self._destination_forward(
            x_dst=x_dst,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return logits, loss

    def _destination_forward(self, x_dst, input_ids, attention_mask):
        x_dst = self.dst_channel_decoder(x_dst)
        decoder_idx, targets, _, is_causal = shift_inputs(
            xb=input_ids,
            attention_mask=attention_mask,
            mode=self.relay_semantic_encoder.mode,
            rate=x_dst.shape[1],
        )
        logits, loss = self.dst_semantic_decoder(
            idx=decoder_idx,
            encoder_output=x_dst,
            is_causal=is_causal,
            enc_padding_mask=None,
            targets=targets,
        )
        return logits, loss

    def _relay_forward(self, x_relay, attention_mask=None):
        x_relay = self.relay_channel_decoder(x_relay)

        # decode every sentence embedding using beam search
        if self.src_semantic_encoder.mode == "sentence":
            return self._relay_forward_sentence(x_relay=x_relay)

        else:
            return self._relay_forward_token(
                x_relay=x_relay,
                attention_mask=attention_mask,
            )

    def _relay_forward_sentence(self, x_relay):
        B, R, C = x_relay.shape

        x_relay = torch.repeat_interleave(input=x_relay, repeats=R, dim=0)
        causal_padding_mask = torch.tril(
            torch.ones(R, R, device=self.device), -1
        ).T.bool()
        causal_padding_mask = causal_padding_mask.repeat(B, 1)
        x_relay, _ = self.relay_semantic_decoder.generate(
            encoder_output=x_relay,
            is_causal=False,
            max_length=self.max_length,  # TODO: fix +1 discrepancy
            enc_padding_mask=causal_padding_mask,
            n_generated_tokens=self.max_length + 1,
        )

        # create attention mask based on [SEP] token
        relay_attention_mask = torch.ones(
            *x_relay.shape, dtype=torch.long, device=self.device
        )
        for i in range(x_relay.shape[0]):
            k = torch.argmax((x_relay[i] == 2).long()).item()
            if k == 0:
                continue
            relay_attention_mask[i, k + 1 :] = 0

        # re-encode decoded sentences and forward
        x_relay = self.relay_semantic_encoder(
            input_ids=x_relay,
            attention_mask=relay_attention_mask,
        )
        x_relay = x_relay.reshape(B, R, C)
        x_relay = self.relay_channel_encoder(x_relay)
        return x_relay

    def _relay_forward_token(self, x_relay, attention_mask):
        x_padding_mask = attention_mask[:, 1:] == 0
        is_causal = True

        x_relay, _ = self.relay_semantic_decoder.generate(
            encoder_output=x_relay,
            is_causal=is_causal,
            max_length=self.max_length,
            enc_padding_mask=x_padding_mask,
            n_generated_tokens=self.max_length + 1,
        )

        # create tril attention mask
        B, T = x_relay.shape
        repeat_amounts = (~x_padding_mask).sum(dim=1)

        relay_attention_mask = torch.tril(
            torch.ones(T, T, device=self.device), diagonal=1
        )
        relay_attention_mask = torch.cat(
            [relay_attention_mask[:i, :] for i in repeat_amounts]
        )

        # re-encode decoded sentences and forward
        x_relay = torch.repeat_interleave(x_relay, repeat_amounts, dim=0)
        x_relay = self.relay_semantic_encoder(
            input_ids=x_relay,
            attention_mask=relay_attention_mask,
        )

        t_idx = torch.cat([torch.arange(i) for i in repeat_amounts])
        x_relay = x_relay[torch.arange(len(t_idx)), t_idx, :]

        # pad the end of sentences
        C = x_relay.shape[-1]
        padded_embeddings = []
        for i in range(B):
            start_idx = repeat_amounts[:i].sum()
            single_sentence = torch.cat(
                [
                    x_relay[start_idx : start_idx + repeat_amounts[i], :],
                    torch.zeros(
                        T - repeat_amounts[i] - 1,
                        C,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                ]
            )
            padded_embeddings.append(single_sentence[None, :, :])

        x_relay = torch.cat(padded_embeddings, dim=0)
        x_relay = self.relay_channel_encoder(x_relay)
        return x_relay

    def _source_forward(self, input_ids, messages, attention_mask):
        x_src = self.src_semantic_encoder(
            messages=messages,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        x_src = self.src_channel_encoder(x_src)
        x_src_to_relay, x_src_to_dst = SemanticTransformer.shift_src_output(
            src_out=x_src,
            mode=self.src_semantic_encoder.mode,
        )
        return x_src_to_dst, x_src_to_relay

    @torch.no_grad()
    def generate(
        self,
        messages: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        d_sd: Optional[float] = None,
        d_sr: Optional[float] = None,
    ):
        # source
        x_src_to_dst, x_src_to_relay = self._source_forward(
            attention_mask=attention_mask,
            input_ids=input_ids,
            messages=messages,
        )

        # relay
        x_relay = self.channel(x_src_to_relay, d_sr)
        x_relay = self._relay_forward(
            x_relay=x_relay,
            attention_mask=attention_mask,
        )

        # destination
        x_dst1 = self.channel(x_relay, d_sd - d_sr)
        x_dst2 = self.channel(x_src_to_dst, d_sd)
        x_dst = torch.cat((x_dst1, x_dst2), dim=-1)

        x_dst = self.dst_channel_decoder(x_dst)

        if self.src_semantic_encoder.mode == "sentence":
            return self.dst_semantic_decoder.generate(
                encoder_output=x_dst,
                is_causal=False,
                max_length=self.max_length,
                enc_padding_mask=None,
                n_generated_tokens=self.max_length + 1,
            )
        else:
            x_padding_mask = attention_mask[:, 1:] == 0
            return self.dst_semantic_decoder.generate(
                encoder_output=x_dst,
                is_causal=True,
                max_length=self.max_length,
                enc_padding_mask=x_padding_mask,
                n_generated_tokens=self.max_length + 1,
            )


def init_relay_semantic_encoder_state_dict(forward_semantic_transformer):
    state_dict = copy.deepcopy(
        forward_semantic_transformer.semantic_encoder.state_dict()
    )
    if "pooling_head" in state_dict:
        state_dict["pooling_head"] = state_dict["pooling_head"][:, [0]]
    return state_dict


def init_dst_channel_decoder_state_dict(forward_semantic_transformer, mode):
    state_dict = copy.deepcopy(
        forward_semantic_transformer.channel_decoder.state_dict()
    )

    if mode != "sentence":
        for i in range(3):
            state_dict[f"layers.{i}.linear.weight"] = state_dict[
                f"layers.{i + 1}.linear.weight"
            ]
            state_dict[f"layers.{i}.linear.bias"] = state_dict[
                f"layers.{i + 1}.linear.bias"
            ]
            state_dict[f"layers.{i}.ln.weight"] = state_dict[
                f"layers.{i + 1}.ln.weight"
            ]
            state_dict[f"layers.{i}.ln.bias"] = state_dict[f"layers.{i + 1}.ln.bias"]

    state_dict["layers.0.linear.weight"] = state_dict["layers.0.linear.weight"].repeat(
        1, 2
    )
    return state_dict


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
