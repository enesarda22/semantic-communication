import numpy as np
import torch
from torch import nn

from semantic_communication.models.semantic_decoder import SemanticDecoder
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.utils.channel import Channel
from semantic_communication.utils.general import get_device
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


class SrcRelayChannelModel(nn.Module):
    def __init__(self, n_in, n_latent, channel: Channel):
        super().__init__()
        self.src_encoder = ChannelEncoder(n_in, n_latent)
        self.relay_decoder = ChannelDecoder(n_latent, n_in)
        self.channel = channel

    def forward(self, src_out, d_sr):
        ch_output = self.channel(src_out, d_sr)
        relay_in = self.relay_decoder(ch_output)
        return relay_in


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


class RelayChannelBlock(nn.Module):
    def __init__(
        self,
        semantic_decoder: SemanticDecoder,
        source_channel_encoder: ChannelEncoder,
        src_relay_channel_model: SrcRelayChannelModel,
    ):
        super().__init__()
        self.source_channel_encoder = source_channel_encoder
        self.src_relay_channel_model = src_relay_channel_model
        self.semantic_decoder = semantic_decoder

    def forward(self, x, d_sr, attention_mask=None, targets=None):
        src_out = self.source_channel_encoder(x)
        relay_in = self.src_relay_channel_model(src_out[:, :-1, :], d_sr)
        logits, loss = self.semantic_decoder(
            encoder_output=relay_in,
            attention_mask=attention_mask,
            targets=targets,
        )

        return logits, loss


class Transceiver(nn.Module):  # TODO: find a cooler name
    def __init__(
        self,
        semantic_encoder: SemanticEncoder,
        relay_channel_block: RelayChannelBlock,
        dst_semantic_decoder: SemanticDecoder,
        src_relay_dst_channel_model: SrcRelayDstChannelModel,
        label_encoder: TensorLabelEncoder,
    ):
        super().__init__()
        self.semantic_encoder = semantic_encoder
        self.source_encoder = relay_channel_block.source_channel_encoder

        self.src_relay_channel_model = relay_channel_block.src_relay_channel_model
        self.relay_semantic_decoder = relay_channel_block.semantic_decoder
        self.relay_encoder = RelayEncoder(
            semantic_encoder=semantic_encoder,
            label_encoder=label_encoder,
        )

        self.dst_semantic_decoder = dst_semantic_decoder
        self.src_relay_dst_channel_model = src_relay_dst_channel_model

    def forward(self, w, attention_mask, targets, d_sd, d_sr, d_rd):
        # transmitter
        encoder_output = self.semantic_encoder(
            input_ids=w,
            attention_mask=attention_mask,
        )
        src_out = self.source_encoder(encoder_output)

        # relay
        relay_in = self.src_relay_channel_model(src_out[:, :-1, :], d_sr)
        logits, _ = self.relay_semantic_decoder(relay_in)
        relay_out = self.relay_encoder(logits)

        # receiver
        receiver_input = self.src_relay_dst_channel_model(
            src_out[:, 1:, :], relay_out, d_rd, d_sd
        )
        receiver_output = self.dst_semantic_decoder(
            encoder_output=receiver_input,
            attention_mask=attention_mask[:, 1:],
            targets=targets,
        )
        return receiver_output


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
