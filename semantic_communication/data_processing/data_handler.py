import csv
from typing import List

import nltk
import torch
from w3lib.html import replace_tags

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import (
    TensorDataset,
    RandomSampler,
    DataLoader,
    SequentialSampler,
)

from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.utils.general import RANDOM_STATE


class DataHandler:
    data_filename = "IMDB Dataset.csv"

    def __init__(
        self,
        semantic_encoder: SemanticEncoder,
        batch_size: int,
        n_samples: int,
        train_size: float,
    ):
        self.semantic_encoder = semantic_encoder

        self.vocab_size = None
        self.encoder = None
        self.train_dataloader = None
        self.val_dataloader = None

        self.batch_size = batch_size
        self.n_samples = n_samples
        self.train_size = train_size

    def load_data(self):
        messages = self.load_text()
        messages = self.preprocess_text(messages)

        tokens = self.semantic_encoder.tokenize(messages=messages)

        self.encoder = LabelEncoder()
        self.encoder.fit(tokens["input_ids"].flatten())
        self.vocab_size = len(self.encoder.classes_)

        input_ids = self.encode_tokens(tokens["input_ids"].flatten())
        attention_mask = tokens["attention_mask"]

        (
            train_input_ids,
            val_input_ids,
            train_attention_mask,
            val_attention_mask,
        ) = train_test_split(
            input_ids,
            attention_mask,
            train_size=self.train_size,
            random_state=RANDOM_STATE,
        )

        train_data = TensorDataset(train_input_ids, train_attention_mask)
        train_sampler = RandomSampler(train_data)
        self.train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=self.batch_size
        )

        val_data = TensorDataset(val_input_ids, val_attention_mask)
        val_sampler = SequentialSampler(val_data)
        self.val_dataloader = DataLoader(
            val_data, sampler=val_sampler, batch_size=self.batch_size
        )

    def load_text(self) -> List[str]:
        with open(self.data_filename, mode="r", encoding="utf-8") as f:
            text = [next(csv.reader(f))[0] for _ in range(self.n_samples + 1)]

        text = text[1:]  # first line is the columns
        return text

    @staticmethod
    def preprocess_text(text: List[str]) -> List[str]:
        sentences_list = [
            nltk.sent_tokenize(replace_tags(m, " ")) for m in text
        ]
        sentences = sum(sentences_list, [])
        return sentences

    def get_tokens(
        self,
        ids,
        attention_mask=None,
        skip_special_tokens=False,
    ) -> List[str]:
        if attention_mask is not None:
            pad_token_id = self.encoder.inverse_transform([0])[0]
            ids = torch.masked_fill(ids, attention_mask == 0, pad_token_id)

        token_ids = self.encoder.inverse_transform(ids.flatten())
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
