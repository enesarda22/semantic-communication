from torch import nn
from torch.nn import functional as F


class AdversarialAgent(nn.Module):
    def __init__(self, n_embeddings):
        super().__init__()
        self.ff_net = nn.Sequential(
            nn.Linear(n_embeddings, 4 * n_embeddings),
            nn.ReLU(),
            nn.Linear(4 * n_embeddings, n_embeddings),
            # TODO: add dropout?
        )

    def forward(self, x):
        x = self.ff_net(x)  # (B, C)
        out = F.normalize(x, p=2, dim=1)
        return out
