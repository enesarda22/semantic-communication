from typing import List

import numpy as np
import torch
from torch import nn
from transformers import BertModel

from semantic_communication.models.semantic_decoder import SemanticDecoder
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.utils.channel import Channel


class ChannelEncComp(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ChannelEncComp, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.linear(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)

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
    # construct the model

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
        for l in self.layers:
            x = l(x)
        return self.linear(x)


class TxRelayChannelModel(nn.Module):
    def __init__(self, nin, n_latent, channel: Channel):
        super(TxRelayChannelModel, self).__init__()

        self.tx_encoder = ChannelEncoder(nin, n_latent)
        self.relay_decoder = ChannelDecoder(n_latent, nin)
        self.channel = channel

    def forward(self, x):
        ch_input = self.tx_encoder(x)
        ch_output = self.channel(ch_input)
        x_hat = self.relay_decoder(ch_output)
        return x_hat


class TxRelayRxChannelModel(nn.Module):
    def __init__(self, nin, n_latent, channel_tx_rx: Channel, channel_rel_rx: Channel):
        super(TxRelayRxChannelModel, self).__init__()

        self.tx_encoder = ChannelEncoder(nin, n_latent)
        self.relay_encoder = ChannelEncoder(nin, n_latent)
        self.rx_decoder = ChannelDecoder(n_latent, nin)
        self.channel_tx_rx = channel_tx_rx
        self.channel_rel_rx = channel_rel_rx

    def forward(self, tx_x, rel_x):
        tx_ch_input = self.tx_encoder(tx_x)
        rel_ch_input = self.relay_encoder(rel_x)

        # Superpose
        ch_output = self.channel_tx_rx(tx_ch_input) + self.channel_rel_rx(rel_ch_input)
        x_hat = self.rx_decoder(ch_output)
        return x_hat  # ground truth = tx_x + rel_x


class Transceiver(nn.Module):  # TODO: find a cooler name
    def __init__(
        self,
        semantic_encoder: SemanticEncoder,
        relay_semantic_decoder: SemanticDecoder,
        rx_semantic_decoder: SemanticDecoder,
        tx_relay_channel_enc_dec: TxRelayChannelModel,
        tx_relay_rx_channel_enc_dec: TxRelayRxChannelModel,
    ):
        super().__init__()
        self.transmitter = Transmitter(semantic_encoder)
        self.relay = Relay(semantic_encoder, relay_semantic_decoder)
        self.receiver = Receiver(rx_semantic_decoder)

        self.tx_relay_channel_enc_dec = tx_relay_channel_enc_dec
        self.tx_relay_rx_channel_enc_dec = tx_relay_rx_channel_enc_dec

    def forward(self, w, attention_mask):
        x = self.transmitter(w)  # B, T, C

        # At relay
        x_hat = self.tx_relay_channel_enc_dec(x[:, :-1, :])
        x_relay = self.relay(x_hat, attention_mask[:, :-1])

        # At receiver
        x_hat_rcv = self.tx_relay_rx_channel_enc_dec(x[:, 1:, :], x_relay)
        s_hat = self.receiver(x_hat_rcv, w[:, 1:], attention_mask[:, 1:])
        return s_hat


class Transmitter(nn.Module):
    def __init__(self, semantic_encoder: SemanticEncoder):
        super().__init__()
        self.semantic_encoder = semantic_encoder

    def forward(self, w: torch.Tensor):
        x = self.semantic_encoder(input_ids=w)
        return x


class Relay(nn.Module):
    def __init__(
        self,
        semantic_encoder: SemanticEncoder,
        semantic_decoder: SemanticDecoder,
    ):
        super().__init__()
        self.semantic_encoder = semantic_encoder
        self.semantic_decoder = semantic_decoder

    def forward(self, x, attention_mask):
        self.semantic_decoder.eval()
        with torch.no_grad():
            predicted_ids = self.semantic_decoder.generate(x, attention_mask)

        begin_padding = torch.full((predicted_ids.shape[0], 1), 1)
        end_padding = torch.full((predicted_ids.shape[0], 1), 2)
        predicted_ids = torch.cat(
            tensors=(begin_padding, predicted_ids, end_padding),
            dim=1,
        )

        out = self.semantic_encoder(input_ids=predicted_ids)
        return out[:, 1:-1, :]


class Receiver(nn.Module):
    def __init__(self, semantic_decoder: SemanticDecoder):
        super().__init__()
        self.semantic_decoder = semantic_decoder

    def forward(self, y, ground_y, attention_mask):
        s = self.semantic_decoder(
            encoder_output=y,
            targets=ground_y,
            attention_mask=attention_mask,
        )
        return s
