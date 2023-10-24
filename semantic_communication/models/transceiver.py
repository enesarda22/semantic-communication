from typing import List

from torch import nn
from transformers import BertModel

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
        # TODO: initialize channel encoder

    def forward(self, s: List[str]):  # TODO: variable name s instead of w?
        x = self.semantic_encoder(s)
        x = self.channel_encoder(x)

        return x


class Relay(nn.Module):
    def __init__(self, semantic_encoder: BertModel):
        super().__init__()
        self.semantic_encoder = semantic_encoder
        # TODO: initialize semantic decoder, channel decoder, channel encoder

    def forward(self, x):
        x = self.channel_decoder(x)  # decode the current token
        s_hat = self.semantic_decoder.generate(x)  # predict next token

        # encoding
        x = self.semantic_encoder(s_hat)
        x1 = self.channel_encoder(x)

        return x1


class Receiver(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: initialize channel decoder, semantic decoder

    def forward(self, y):
        y = self.channel_decoder(y)
        s = self.semantic_decoder(y)

        return s


# TODO: channel encoder decoder goes here
