from typing import List, Optional

import torch
from torch import nn

from transformers import AutoTokenizer, AutoModel

from semantic_communication.utils.general import get_device


class SemanticEncoder(nn.Module):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, label_encoder, max_length, mode, rate=None):
        super().__init__()
        self.device = get_device()

        self.label_encoder = label_encoder
        self.max_length = max_length + 1  # TODO: fix +1 discrepancy
        self.mode = mode
        self.rate = rate

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.bert = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.bert.embeddings.word_embeddings.weight = nn.Parameter(
            self.bert.embeddings.word_embeddings.weight[label_encoder.classes, :]
        )

    def forward(
        self,
        messages: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        if messages is not None:
            tokens = self.tokenize(messages=messages, label_encoder=True)
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]

        encoder_lhs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )["last_hidden_state"]

        if self.mode == "predict":
            encoder_output = self.mean_pooling(
                bert_lhs=encoder_lhs,
                attention_mask=attention_mask,
            )
            encoder_output = torch.cat(
                tensors=(encoder_output.unsqueeze(1), encoder_lhs[:, 1:, :]),
                dim=1,
            )
        elif self.mode == "forward":
            encoder_output = encoder_lhs[:, 1:, :]
        elif self.mode == "sentence" or self.mode == "next_sentence":
            encoder_output = self.mean_pooling(
                bert_lhs=encoder_lhs,
                attention_mask=attention_mask,
            )
            encoder_output = encoder_output[:, None, :]
        else:
            raise ValueError("Mode needs to be 'predict', 'forward' or 'sentence'.")

        return encoder_output

    def tokenize(self, messages: List[str], label_encoder=False):
        tokens = self.tokenizer(
            messages,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        if label_encoder:
            tokens["input_ids"] = self.label_encoder.transform(tokens["input_ids"])

        return tokens

    def get_tokens(
        self,
        ids=None,
        token_ids=None,
        attention_mask=None,
        skip_special_tokens=False,
    ) -> List[str]:
        if token_ids is None:
            token_ids = self.label_encoder.inverse_transform(ids)

        if attention_mask is not None:
            token_ids = torch.masked_fill(token_ids, attention_mask == 0, 0)

        tokens = [
            self.tokenizer.decode(t, skip_special_tokens=skip_special_tokens)
            for t in token_ids
        ]
        return tokens

    @staticmethod
    def mean_pooling(bert_lhs, attention_mask=None):
        if attention_mask is None:
            out = torch.mean(bert_lhs, 1)

        else:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(bert_lhs.size()).float()
            )

            out = torch.sum(bert_lhs * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

        return out
