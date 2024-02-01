import torch

from semantic_communication.utils.general import get_device


class TensorLabelEncoder:
    def __init__(self):
        self.classes = None
        self.pad_id = None
        self.cls_id = None
        self.device = get_device()

    def fit(self, x):
        self.classes = torch.unique(x)
        self.pad_id = torch.nonzero(self.classes == 0)[0][0].item()
        self.cls_id = torch.nonzero(self.classes == 101)[0][0].item()
        return self

    def transform(self, x):
        bool_tensor = torch.eq(
            x.reshape(1, -1), self.classes.reshape(-1, 1).to(self.device)
        )
        indices = torch.argmax(bool_tensor.to(torch.int8), dim=0)
        return indices.view(x.shape)

    def inverse_transform(self, y):
        return self.classes[y]
