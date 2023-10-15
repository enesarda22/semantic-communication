import csv
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModel


class DataHandler:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    max_length = 10

    def __init__(self, device):
        self.device = device

        self.tokenizer = None
        self.vocab_size = None
        self.train_dataloader = None
        self.val_dataloader = None

        self.batch_size = 32

        self.train_size = 0.8

    def load_data(self):
        text = []
        with open("IMDB Dataset.csv", mode="r") as file:
            csv_reader = csv.reader(file)
            for line in csv_reader:
                text.append(line[0])
        text = text[1:]

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.vocab_size = self.tokenizer.vocab_size

        tokens = self.tokenizer(
            text[:1000],
            padding="max_length",
            max_length=self.max_length + 2,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        bert = AutoModel.from_pretrained(self.model_name).to(self.device)
        bert.eval()
        with torch.no_grad():
            encoder_output = bert(**tokens)

        encoder_output = self.mean_pooling(
            model_output=encoder_output,
            attention_mask=tokens["attention_mask"],
        )

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
            input_ids, encoder_output, train_size=self.train_size
        )  # tokens["input_ids"]

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
    def mean_pooling(model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
