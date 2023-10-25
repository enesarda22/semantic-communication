from typing import List

import numpy as np
import torch
from torch import nn
from transformers import BertModel

from semantic_communication.models.semantic_decoder import SemanticDecoder
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.utils.channel import Channel


class Transceiver(nn.Module):  # TODO: find a cooler name
    def __init__(
        self,
        tx_relay_ch: Channel,
        relay_rcv_ch: Channel,
        tx_rcv_ch: Channel,
    ):
        super().__init__()
        self.tx_relay_ch = tx_relay_ch
        self.relay_rcv_ch = relay_rcv_ch
        self.tx_rcv_ch = tx_rcv_ch

        self.transmitter = Transmitter()
        self.relay = Relay()
        self.receiver = Receiver()

    def forward(self, s):
        x = self.transmitter(s)  # B, T, C

        y1 = self.tx_relay_ch(x[:, :-1, :])
        y2 = self.relay_rcv_ch(self.relay(y1))
        y3 = self.tx_rcv_ch(x[:, 1:, :])
        y4 = y2 + y3  # superposition

        s_hat = self.receiver(y4)
        return s_hat


class Transmitter(nn.Module):
    def __init__(self, semantic_encoder: BertModel):
        super().__init__()
        self.semantic_encoder = semantic_encoder
        self.channel_encoder = ChannelEncoder(384, 128)

    def forward(self, s: List[str]):  # TODO: variable name s instead of w?
        x = self.semantic_encoder(s)
        x = self.channel_encoder(x)

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

    def forward(self, x):
        self.semantic_decoder.eval()
        with torch.no_grad():
            predicted_ids = self.semantic_decoder.generate(x)

        begin_padding = torch.full((predicted_ids.shape[0], 1), 1)
        end_padding = torch.full((predicted_ids.shape[0], 1), 2)
        predicted_ids = torch.cat(
            tensors=(begin_padding, predicted_ids, end_padding),
            dim=1,
        )

        out = self.semantic_encoder(input_ids=predicted_ids)
        return out[:, 1:-1, :]


class Receiver(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel_decoder = ChannelDecoder(128, 384)
        # TODO: initialize semantic decoder

    def forward(self, y):
        y = self.channel_decoder(y)
        s = self.semantic_decoder(y)

        return s


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
