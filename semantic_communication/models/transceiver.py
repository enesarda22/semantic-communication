import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn

from semantic_communication.models.semantic_decoder import SemanticDecoder
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.utils.channel import Channel
from semantic_communication.utils.general import get_device


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


class TxRelayChannelModel(nn.Module):
    def __init__(self, nin, n_latent, channel: Channel):
        super(TxRelayChannelModel, self).__init__()
        self.relay_decoder = ChannelDecoder(n_latent, nin)
        self.channel = channel

    def forward(self, x, d_sr):
        ch_output = self.channel(x, d_sr)
        x_hat = self.relay_decoder(ch_output)
        return x_hat


class TxRelayRxChannelModel(nn.Module):
    def __init__(
        self,
        nin,
        n_latent,
        channel: Channel,
    ):
        super(TxRelayRxChannelModel, self).__init__()

        self.tx_encoder = ChannelEncoder(nin, n_latent)
        self.relay_encoder = ChannelEncoder(nin, n_latent)
        self.rx_decoder = ChannelDecoder(n_latent * 2, nin * 2)
        self.channel = channel

    def forward(self, tx_x, rel_x, d_rd, d_sd):
        rel_ch_input = self.relay_encoder(rel_x)
        tx_ch_input = self.tx_encoder(tx_x)

        rel_ch_out = self.channel(rel_ch_input, d_rd)
        tx_ch_out = self.channel(tx_ch_input, d_sd)

        ch_output = torch.cat([rel_ch_out, tx_ch_out], dim=-1)  # concatenate
        x_hat = self.rx_decoder(ch_output)
        return x_hat


class RelayChannelBlock(nn.Module):
    def __init__(
        self,
        semantic_decoder: SemanticDecoder,
        tx_channel_enc: ChannelEncoder,
        tx_relay_channel_enc_dec: TxRelayChannelModel,
    ):
        super().__init__()
        self.semantic_decoder = semantic_decoder
        self.tx_channel_enc = tx_channel_enc
        self.tx_relay_channel_enc_dec = tx_relay_channel_enc_dec

    def forward(self, x, d_sr, attention_mask=None, targets=None):
        tx_out = self.tx_channel_enc(x)
        relay_in = self.tx_relay_channel_enc_dec(tx_out[:, :-1, :], d_sr)
        logits, loss = self.semantic_decoder(
            encoder_output=relay_in,
            attention_mask=attention_mask,
            targets=targets,
        )

        return tx_out, logits, loss


class Transceiver(nn.Module):  # TODO: find a cooler name
    def __init__(
        self,
        semantic_encoder: SemanticEncoder,
        relay_channel_block: RelayChannelBlock,
        rx_semantic_decoder: SemanticDecoder,
        tx_relay_rx_channel_enc_dec: TxRelayRxChannelModel,
        encoder: LabelEncoder,
    ):
        super().__init__()
        self.tx_semantic_encoder = semantic_encoder
        self.relay = Relay(semantic_encoder, relay_channel_block, encoder)
        self.rx_semantic_decoder = rx_semantic_decoder
        self.tx_relay_rx_channel_enc_dec = tx_relay_rx_channel_enc_dec

    def forward(self, w, attention_mask, targets, d_sd, d_sr, d_rd):
        # transmitter
        encoder_output = self.tx_semantic_encoder(
            input_ids=w,
            attention_mask=attention_mask,
        )

        # relay
        source_output, relay_output = self.relay(encoder_output, d_sr)

        # receiver
        receiver_input = self.tx_relay_rx_channel_enc_dec(
            source_output[:, 1:, :], relay_output, d_rd, d_sd
        )
        receiver_output = self.rx_semantic_decoder(
            encoder_output=receiver_input,
            attention_mask=attention_mask[:, 1:],
            targets=targets,
        )
        return receiver_output


class Relay(nn.Module):
    def __init__(
        self,
        semantic_encoder: SemanticEncoder,
        relay_channel_block: RelayChannelBlock,
        encoder: LabelEncoder,
    ):
        super().__init__()
        self.device = get_device()
        self.semantic_encoder = semantic_encoder
        self.relay_channel_block = relay_channel_block
        self.encoder = encoder

    def forward(self, x, d_sr):
        B, T, C = x.shape

        tx_out, logits, _ = self.relay_channel_block(x, d_sr)
        predicted_ids = torch.argmax(logits, dim=-1)

        predicted_ids = self.encoder.inverse_transform(
            predicted_ids.flatten().to("cpu")
        ).reshape(B, T)
        predicted_ids = torch.LongTensor(predicted_ids).to(self.device)

        # ids are repeated to generate the embeddings sequentially
        predicted_ids = torch.repeat_interleave(predicted_ids, T, dim=0)

        # append [CLS] token
        cls_padding = torch.full((B * T, 1), 101).to(self.device)
        predicted_ids = torch.cat(
            tensors=(cls_padding, predicted_ids),
            dim=1,
        )

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
        out = out.view(B, T, C)

        return tx_out, out
