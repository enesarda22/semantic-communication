import os
from typing import List

import torch

from torch.utils.data import (
    RandomSampler,
    DataLoader,
)

from semantic_communication.data_processing.preprocessor import Preprocessor
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.utils.general import get_device


class DataHandler:
    def __init__(
        self,
        semantic_encoder: SemanticEncoder,
        data_fp: str,
        batch_size: int,
    ):
        self.device = get_device()
        self.semantic_encoder = semantic_encoder
        self.data_fp = data_fp
        self.batch_size = batch_size

        encoder_fp = os.path.join(data_fp, Preprocessor.encoder_fn)
        self.encoder = torch.load(encoder_fp, map_location="cpu")

        self.vocab_size = len(self.encoder.classes_)

        self.train_dataloader = self.init_dl(fn=Preprocessor.train_data_fn)
        self.val_dataloader = self.init_dl(fn=Preprocessor.val_data_fn)
        self.test_dataloader = self.init_dl(fn=Preprocessor.test_data_fn)

    def get_tokens(
        self,
        ids,
        attention_mask=None,
        skip_special_tokens=False,
    ) -> List[str]:
        if attention_mask is not None:
            pad_token_id = self.encoder.transform([0])[0]
            ids = torch.masked_fill(ids, attention_mask == 0, pad_token_id)

        token_ids = self.encoder.inverse_transform(ids.flatten().to("cpu"))
        token_ids = token_ids.reshape(ids.shape)

        tokens = [
            self.semantic_encoder.tokenizer.decode(
                t, skip_special_tokens=skip_special_tokens
            )
            for t in token_ids
        ]
        return tokens

    def encode_token_ids(self, token_ids: torch.Tensor):
        ids = self.encoder.transform(token_ids.flatten().to("cpu"))
        ids = ids.reshape(-1, self.semantic_encoder.max_length)
        return torch.LongTensor(ids).to(self.device)

    def init_dl(self, fn: str):
        fp = os.path.join(self.data_fp, fn)

        dataset = torch.load(fp, map_location=self.device)
        sampler = RandomSampler(dataset)

        return DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=self.batch_size,
        )
