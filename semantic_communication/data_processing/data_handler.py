import os
import pickle
from typing import List

import torch

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import (
    TensorDataset,
    RandomSampler,
    DataLoader,
)

from semantic_communication.data_processing.preprocessor import Preprocessor
from semantic_communication.models.semantic_encoder import SemanticEncoder


class DataHandler:
    def __init__(
        self,
        semantic_encoder: SemanticEncoder,
        data_fp: str,
        batch_size: int,
    ):
        self.semantic_encoder = semantic_encoder

        self.vocab_size = None
        self.encoder = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

        self.data_fp = data_fp
        self.batch_size = batch_size

    def load_data(self):
        self.train_dataloader = self.init_dl(fn=Preprocessor.train_data_fn)
        self.val_dataloader = self.init_dl(fn=Preprocessor.val_data_fn)
        self.test_dataloader = self.init_dl(fn=Preprocessor.test_data_fn)

    @staticmethod
    def read_data(fp):
        with open(fp, "rb") as f:
            data = pickle.load(f)

        return data

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

    def encode_tokens(self, tokens):
        ids = self.encoder.transform(tokens)
        ids = ids.reshape(-1, self.semantic_encoder.max_length)
        return torch.LongTensor(ids)

    def init_dl(self, fn: str):
        fp = os.path.join(self.data_fp, fn)
        messages = self.read_data(fp=fp)
        tokens = self.semantic_encoder.tokenize(messages=messages)

        self.encoder = LabelEncoder()
        self.encoder.fit(tokens["input_ids"].flatten().to("cpu"))
        self.vocab_size = len(self.encoder.classes_)

        input_ids = self.encode_tokens(tokens["input_ids"].flatten().to("cpu"))
        attention_mask = tokens["attention_mask"]

        dataset = TensorDataset(input_ids, attention_mask)
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=self.batch_size,
        )

        return dataloader
