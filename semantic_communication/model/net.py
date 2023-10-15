from torch import nn
import torch


class Net(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        feedforward_dim: int,
        dropout: float,
        target_vocab_size: int,
        num_layers: int,
        max_length: int,
    ):
        super().__init__()

        self.word_embedding = nn.Embedding(target_vocab_size, embedding_dim)

        # TODO: change positional embedding to use sine and cosine
        self.positional_embedding = nn.Embedding(max_length, embedding_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.fc_out = nn.Linear(embedding_dim, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output):
        N, sequence_length = x.shape
        positions = torch.arange(0, sequence_length).expand(N, sequence_length)

        x = self.dropout(self.word_embedding(x) + self.positional_embedding(positions))
        x = self.decoder(
            tgt=x,
            memory=encoder_output.unsqueeze(1).expand(N, sequence_length, 384),
            tgt_mask=self.get_attention_mask(sequence_length),
        )

        out = self.fc_out(x)
        return out

    @staticmethod
    def get_attention_mask(seq_length):
        return torch.tril(torch.ones(seq_length, seq_length), diagonal=-1).T.bool()
