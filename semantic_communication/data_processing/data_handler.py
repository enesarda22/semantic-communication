import csv
from typing import List

import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModel


class DataHandler:
    data_filename = "IMDB Dataset.csv"
    max_length = 10
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, device):
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.vocab_size = self.tokenizer.vocab_size

        self.encoder = None
        self.train_dataloader = None
        self.val_dataloader = None

        self.batch_size = 32
        self.n_samples = 40000
        self.train_size = 0.8

    def load_data(self):
        text = self.load_text()
        tokens = self.tokenize(text=text)
        encoder_output = self.load_encoder_output(tokens=tokens)

        self.encoder = LabelEncoder()
        input_ids = self.encoder.fit_transform(tokens["input_ids"][:, :-1].flatten())
        input_ids = torch.LongTensor(input_ids.reshape(-1, self.max_length + 1))
        self.vocab_size = len(self.encoder.classes_)

        (
            train_input_ids,
            val_input_ids,
            train_encoder_outputs,
            val_encoder_outputs,
        ) = train_test_split(
            input_ids,
            encoder_output,
            train_size=self.train_size,
            random_state=42,
        )

        train_data = TensorDataset(train_input_ids, train_encoder_outputs)
        train_sampler = RandomSampler(train_data)
        self.train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=self.batch_size
        )

        val_data = TensorDataset(val_input_ids, val_encoder_outputs)
        val_sampler = SequentialSampler(val_data)
        self.val_dataloader = DataLoader(
            val_data, sampler=val_sampler, batch_size=self.batch_size
        )

    @staticmethod
    def mean_pooling(bert_lhs, attention_mask):
        token_embeddings = bert_lhs
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    @classmethod
    def load_text(cls) -> List[str]:
        text = []
        with open(cls.data_filename, mode="r") as file:
            csv_reader = csv.reader(file)
            for line in csv_reader:
                text.append(line[0])
        text = text[1:]

        return text

    def tokenize(self, text: List[str]):
        return self.tokenizer(
            text[: self.n_samples],
            padding="max_length",
            max_length=self.max_length + 2,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

    def load_encoder_output(self, tokens):
        bert = AutoModel.from_pretrained(self.model_name).to(self.device)
        bert.eval()
        with torch.no_grad():
            bert_output = bert(**tokens)

        bert_lhs = bert_output["last_hidden_state"]
        mean_pooling_output = self.mean_pooling(
            bert_lhs=bert_lhs,
            attention_mask=tokens["attention_mask"],
        )
        encoder_output = torch.cat(
            tensors=(mean_pooling_output.unsqueeze(1), bert_lhs[:, 1:, :]),
            dim=1,
        )

        return encoder_output

    def get_tokens(self, ids):
        token_ids = self.encoder.inverse_transform(ids.flatten())
        token_ids = token_ids.reshape(ids.shape)

        tokens = [self.tokenizer.decode(t) for t in token_ids]
        return tokens
