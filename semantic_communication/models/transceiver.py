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
from semantic_communication.utils.general import get_device, shift_inputs, pad_cls


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

        if self.src_semantic_encoder.mode == "next_sentence":
            input_ids = pad_cls(input_ids[:, -self.max_length :])

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

    def _relay_forward(self, x_relay, attention_mask=None):
        self.relay_channel_decoder.eval()
        x_relay = self.relay_channel_decoder(x_relay)

        # decode every sentence embedding using beam search
        if (
            self.src_semantic_encoder.mode == "sentence"
            or self.src_semantic_encoder.mode == "next_sentence"
        ):
            return self._relay_forward_sentence(x_relay=x_relay)

        else:
            return self._relay_forward_token(
                x_relay=x_relay,
                attention_mask=attention_mask,
            )

    def _relay_forward_sentence(self, x_relay):
        B, R, C = x_relay.shape

        self.relay_semantic_decoder.eval()
        x_relay, _ = self.relay_semantic_decoder.generate(
            encoder_output=x_relay,
            is_causal=False,
            max_length=self.max_length,  # TODO: fix +1 discrepancy
            enc_padding_mask=None,
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

        self.relay_semantic_decoder.eval()
        x_relay = self.relay_semantic_decoder.generate_greedy(
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
            torch.ones(T, T, device=self.device, dtype=torch.int64), diagonal=1
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
        self.src_semantic_encoder.eval()
        x_src = self.src_semantic_encoder(
            messages=messages,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        self.src_channel_encoder.eval()
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
        greedy: bool = False,
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

        if (
            self.src_semantic_encoder.mode == "sentence"
            or self.src_semantic_encoder.mode == "next_sentence"
        ):
            if greedy:
                return self.dst_semantic_decoder.generate_greedy(
                    encoder_output=x_dst,
                    is_causal=False,
                    max_length=self.max_length,
                    enc_padding_mask=None,
                    n_generated_tokens=self.max_length + 1,
                )
            else:
                return self.dst_semantic_decoder.generate(
                    encoder_output=x_dst,
                    is_causal=False,
                    max_length=self.max_length,
                    enc_padding_mask=None,
                    n_generated_tokens=self.max_length + 1,
                )
        else:
            x_padding_mask = attention_mask[:, 1:] == 0
            if greedy:
                return self.dst_semantic_decoder.generate_greedy(
                    encoder_output=x_dst,
                    is_causal=True,
                    max_length=self.max_length,
                    enc_padding_mask=x_padding_mask,
                    n_generated_tokens=self.max_length + 1,
                )
            else:
                return self.dst_semantic_decoder.generate(
                    encoder_output=x_dst,
                    is_causal=True,
                    max_length=self.max_length,
                    enc_padding_mask=x_padding_mask,
                    n_generated_tokens=self.max_length + 1,
                )


def init_src_relay_transformer_from_transceiver(state_dict_path):
    cp = torch.load(state_dict_path, map_location=get_device())
    replace_keys = {
        "src_semantic_encoder": "semantic_encoder",
        "relay_semantic_decoder": "semantic_decoder",
        "src_channel_encoder": "channel_encoder",
        "relay_channel_decoder": "channel_decoder",
    }

    state_dict = {}
    for replace_key, replace_value in replace_keys.items():
        wanted_keys = [k for k in cp["model_state_dict"].keys() if replace_key in k]
        updated_state_dict = {
            k.replace(replace_key, replace_value): cp["model_state_dict"][k]
            for k in wanted_keys
        }
        state_dict.update(updated_state_dict)

    return state_dict


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

    if mode != "sentence" and mode != "next_sentence":
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
