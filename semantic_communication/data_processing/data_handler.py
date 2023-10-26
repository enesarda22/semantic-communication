import csv
from typing import List

import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler

from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.utils.general import get_device, RANDOM_STATE


class DataHandler:
    data_filename = "IMDB Dataset.csv"

    def __init__(
        self,
        semantic_encoder: SemanticEncoder,
        batch_size: int,
        n_samples: int,
        train_size: float,
    ):
        self.device = get_device()

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
        tokens = self.semantic_encoder.tokenize(messages=messages)

        encoder_output = self.semantic_encoder(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
        )

        self.encoder = LabelEncoder()
        self.encoder.fit(tokens["input_ids"][:, :-1].flatten())
        self.vocab_size = len(self.encoder.classes_)

        input_ids = self.encode_tokens(tokens["input_ids"][:, :-1].flatten())

        (
            train_input_ids,
            val_input_ids,
            train_encoder_outputs,
            val_encoder_outputs,
        ) = train_test_split(
            input_ids,
            encoder_output,
            train_size=self.train_size,
            random_state=RANDOM_STATE,
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

    def load_text(self) -> List[str]:
        text = []
        with open(self.data_filename, mode="r", encoding="utf-8") as file:
            csv_reader = csv.reader(file)
            for i, line in enumerate(csv_reader):
                if i > self.n_samples:
                    break
                text.append(line[0])

        text = text[1:]
        return text

    # TODO: remove
    # def load_encoder_output(self, tokens):
    #     bert.eval()
    #     with torch.no_grad():
    #         bert_output = bert(**tokens)
    #
    #     bert_lhs = bert_output["last_hidden_state"]
    #     mean_pooling_output = self.mean_pooling(
    #         bert_lhs=bert_lhs,
    #         attention_mask=tokens["attention_mask"],
    #     )
    #     encoder_output = torch.cat(
    #         tensors=(mean_pooling_output.unsqueeze(1), bert_lhs[:, 1:, :]),
    #         dim=1,
    #     )
    #
    #     return encoder_output

    def get_tokens(self, ids):
        token_ids = self.encoder.inverse_transform(ids.flatten())
        token_ids = token_ids.reshape(ids.shape)

        tokens = [self.semantic_encoder.tokenizer.decode(t) for t in token_ids]
        return tokens

    def encode_tokens(self, tokens):
        ids = self.encoder.transform(tokens)
        ids = torch.LongTensor(ids.reshape(-1, self.semantic_encoder.max_length + 1))

        return ids