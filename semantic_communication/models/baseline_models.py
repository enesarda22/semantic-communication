from torch import nn
from semantic_communication.utils.channel import Channel
from semantic_communication.models.transceiver import ChannelEncoder, ChannelDecoder
import torch
import torch.nn.functional as F


class Tx_Relay(nn.Module):
    def __init__(self, nin, n_emb, n_latent, channel: Channel, entire_network_train=0):
        super(Tx_Relay, self).__init__()

        self.embedding_layer = nn.Embedding(nin, n_emb)
        self.tx_encoder = ChannelEncoder(n_emb, n_latent)
        self.relay_decoder = ChannelDecoder(n_latent, n_emb)
        self.linear = nn.Linear(n_emb, nin)
        self.channel = channel
        self.entire_network_train = entire_network_train

    def forward(self, x, attention_mask, d_sr):
        embeddings = self.embedding_layer(x)
        ch_input = self.tx_encoder(embeddings)

        if self.channel is not None:
            ch_output = self.channel(ch_input, d_sr)
        else:
            last_dim = ch_input.shape[-1]
            ch_output = (
                (self.signal_power_constraint * last_dim / 2) ** 0.5
            ) * F.normalize(ch_input, dim=-1, p=2)

        x_hat = self.linear(self.relay_decoder(ch_output))

        if self.entire_network_train == 0:
            B, T, C = x_hat.shape
            logits = x_hat.reshape(B * T, C)
            grnd_x = x.reshape(B * T)
            attention_mask = attention_mask.flatten() == 1

            loss = F.cross_entropy(logits[attention_mask, :], grnd_x[attention_mask])

            return x_hat, ch_input, loss

        else:
            return x_hat, ch_input


class Tx_Relay_Rx(nn.Module):
    def __init__(
        self, nin, n_emb, n_latent, channel: Channel, tx_relay_model: Tx_Relay
    ):
        super(Tx_Relay_Rx, self).__init__()

        self.tx_relay_model = tx_relay_model
        # Freeze
        for param in self.tx_relay_model.parameters():
            param.requires_grad = False

        self.relay_embedding = nn.Embedding(nin, n_emb)
        self.relay_encoder = ChannelEncoder(n_emb, n_latent)
        self.rx_decoder_1 = ChannelDecoder(n_latent, n_emb)
        self.rx_decoder_2 = ChannelDecoder(n_latent, n_emb)

        self.linear = nn.Linear(int(2 * n_emb), nin)
        self.channel = channel

    def forward(self, x, attention_mask, d_sd, d_sr, d_rd):
        x_hat, x1 = self.tx_relay_model(x, attention_mask, d_sr)

        x_hard = torch.argmax(x_hat, dim=2)

        x_emb = self.relay_embedding(x_hard)

        y2 = self.channel(self.relay_encoder(x_emb), d_rd)
        y1 = self.channel(x1, d_sd)

        x_hat = torch.cat((self.rx_decoder_1(y1), self.rx_decoder_2(y2)), dim=2)
        x_hat = self.linear(x_hat)

        B, T, C = x_hat.shape
        logits = x_hat.reshape(B * T, C)
        grnd_x = x.reshape(B * T)
        attention_mask = attention_mask.flatten() == 1

        loss = F.cross_entropy(logits[attention_mask, :], grnd_x[attention_mask])
        return x_hat, loss
