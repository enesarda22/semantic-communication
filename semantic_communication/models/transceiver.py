from typing import List

import torch
from torch import nn
from transformers import BertModel

from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.models.semantic_decoder import SemanticDecoder
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
    def __init__(
        self,
        semantic_encoder: BertModel,
        semantic_decoder: SemanticDecoder,
    ):
        super().__init__()
        self.semantic_encoder = semantic_encoder
        self.semantic_decoder = semantic_decoder

    def forward(self, x):
        self.semantic_decoder.eval()
        with torch.no_grad():
            predicted_ids = self.semantic_decoder.generate(x)

        begin_padding = torch.ones((predicted_ids.shape[0], 1), dtype=torch.long)
        end_padding = 2 * torch.ones((predicted_ids.shape[0], 1), dtype=torch.long)
        predicted_ids = torch.cat((begin_padding, predicted_ids, end_padding), dim=1)

        relay_output = self.semantic_encoder(input_ids=predicted_ids)
        bert_lhs = relay_output["last_hidden_state"]

        mean_pooling_out = DataHandler.mean_pooling(
            bert_lhs=bert_lhs,
            attention_mask=torch.ones(bert_lhs.shape[:-1]),
        )

        out = torch.cat(
            tensors=(mean_pooling_out.unsqueeze(1), bert_lhs[:, 1:, :]),
            dim=1,
        )
        return out[:, 1:-1, :]


class Receiver(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: initialize channel decoder, semantic decoder

    def forward(self, y):
        y = self.channel_decoder(y)
        s = self.semantic_decoder(y)

        return s


class ChannelDecoder(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

    def forward(self):
        pass


class ChannelEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

    def forward(self):
        pass
