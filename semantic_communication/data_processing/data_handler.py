import os

import torch

from torch.utils.data import (
    RandomSampler,
    DataLoader,
    DistributedSampler,
    SequentialSampler,
)

from semantic_communication.data_processing.preprocessor import Preprocessor
from semantic_communication.utils.general import get_device, valid_mode


class DataHandler:
    def __init__(
        self,
        data_fp: str,
        batch_size: int,
        mode: str,
        rank=None,
        world_size=None,
    ):
        assert valid_mode(mode)

        self.device = get_device()
        self.data_fp = data_fp
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size

        if mode == "next_sentence":
            label_encoder_fp = os.path.join(
                data_fp, Preprocessor.next_sentence_pred_encoder_fn
            )
            self.label_encoder = torch.load(label_encoder_fp, map_location=self.device)

            self.vocab_size = len(self.label_encoder.classes)

            self.train_dataloader = self.init_dl(
                fn=Preprocessor.next_sentence_pred_train_data_fn
            )
            self.val_dataloader = self.init_dl(
                fn=Preprocessor.next_sentence_pred_val_data_fn
            )
            self.test_dataloader = self.init_dl(
                fn=Preprocessor.next_sentence_pred_test_data_fn, random=False
            )
        else:
            label_encoder_fp = os.path.join(data_fp, Preprocessor.encoder_fn)
            self.label_encoder = torch.load(label_encoder_fp, map_location=self.device)

            self.vocab_size = len(self.label_encoder.classes)

            self.train_dataloader = self.init_dl(fn=Preprocessor.train_data_fn)
            self.val_dataloader = self.init_dl(fn=Preprocessor.val_data_fn)
            self.test_dataloader = self.init_dl(
                fn=Preprocessor.test_data_fn, random=False
            )

    def init_dl(self, fn: str, random=True):
        fp = os.path.join(self.data_fp, fn)

        dataset = torch.load(fp, map_location=self.device)

        if self.rank is None or self.world_size is None:
            if random:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)
        else:
            sampler = DistributedSampler(
                dataset=dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=random,
                drop_last=False,
            )

        return DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=self.batch_size,
        )
