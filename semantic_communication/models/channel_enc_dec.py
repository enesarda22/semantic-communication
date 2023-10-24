import numpy as np
import torch.nn as nn
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Channel_Enc_Comp(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Channel_Enc_Comp, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.linear(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)

        out = self.prelu(x)
        return out


class Channel_Encoder(nn.Module):
    def __init__(self, nin, nout):
        super(Channel_Encoder, self).__init__()
        up_dim = int(np.floor(np.log2(nin) / 2))
        low_dim = int(np.ceil(np.log2(nout) / 2))

        dims = [nin]
        for i in range(up_dim - low_dim + 1):
            dims.append(np.power(4, up_dim - i))

        self.layers = nn.ModuleList(
            [Channel_Enc_Comp(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        self.linear = nn.Linear(dims[-1], nout)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return self.linear(x)


class Channel_Decoder(nn.Module):
    # construct the model

    def __init__(self, nin, nout):
        super(Channel_Decoder, self).__init__()
        up_dim = int(np.floor(np.log2(nout) / 2))
        low_dim = int(np.ceil(np.log2(nin) / 2))
        dims = [nin]
        for i in range(up_dim - low_dim + 1):
            dims.append(np.power(4, low_dim + i))

        self.layers = nn.ModuleList(
            [Channel_Enc_Comp(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        self.linear = nn.Linear(dims[-1], nout)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return self.linear(x)
