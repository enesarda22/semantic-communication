import os

import torch

from torch.utils.data import (
    RandomSampler,
    DataLoader,
)

from semantic_communication.data_processing.preprocessor import Preprocessor
from semantic_communication.utils.general import get_device


class DataHandler:
    def __init__(
        self,
        data_fp: str,
        batch_size: int,
    ):
        self.device = get_device()
        self.data_fp = data_fp
        self.batch_size = batch_size

        label_encoder_fp = os.path.join(data_fp, Preprocessor.encoder_fn)
        self.label_encoder = torch.load(label_encoder_fp, map_location=self.device)

        self.vocab_size = len(self.label_encoder.classes)

        self.train_dataloader = self.init_dl(fn=Preprocessor.train_data_fn)
        self.val_dataloader = self.init_dl(fn=Preprocessor.val_data_fn)
        self.test_dataloader = self.init_dl(fn=Preprocessor.test_data_fn)

    def init_dl(self, fn: str):
        fp = os.path.join(self.data_fp, fn)

        dataset = torch.load(fp, map_location=self.device)
        sampler = RandomSampler(dataset)

        return DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=self.batch_size,
        )
