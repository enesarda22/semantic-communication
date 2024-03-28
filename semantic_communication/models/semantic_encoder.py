from typing import List, Optional

import torch
from torch import nn

from transformers import AutoTokenizer, AutoModel

from semantic_communication.utils.general import get_device


class SemanticEncoder(nn.Module):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __init__(self, label_encoder, max_length, mode):
        super().__init__()
        self.device = get_device()

        self.label_encoder = label_encoder
        self.max_length = max_length + 1
        self.mode = mode

        self.bert = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.bert.embeddings.word_embeddings.weight = nn.Parameter(
            self.bert.embeddings.word_embeddings.weight[label_encoder.classes, :]
        )

    def forward(
        self,
        messages: Optional[List[str]] = None,
        m1: Optional[List[str]] = None,
        m2: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        if messages is not None or (m1 is not None and m2 is not None):
            tokens = self.tokenize(
                messages=messages,
                m1=m1,
                m2=m2,
                max_length=self.max_length,
                device=self.device,
            )
            input_ids = self.label_encoder.transform(tokens["input_ids"])
            attention_mask = tokens["attention_mask"]

        encoder_lhs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )["last_hidden_state"]

        if self.mode == "predict":
            encoder_output = self._replace_cls_embedding(
                encoder_lhs=encoder_lhs,
                attention_mask=attention_mask,
            )
            encoder_output = encoder_output[:, :-1, :]
        elif self.mode == "sentence":
            encoder_output = self._replace_cls_embedding(
                encoder_lhs=encoder_lhs,
                attention_mask=attention_mask,
            )
            encoder_output = encoder_output[:, [0], :]
        elif self.mode == "forward":
            encoder_output = encoder_lhs[:, 1:, :]
        elif self.mode == "cls":
            encoder_output = encoder_lhs[:, [0], :]
        else:
            raise ValueError(
                "Mode needs to be 'predict', 'forward', 'sentence' or 'cls'."
            )

        return encoder_output

    @classmethod
    def tokenize(
        cls,
        messages: Optional[List[str]] = None,
        m1: Optional[List[str]] = None,
        m2: Optional[List[str]] = None,
        max_length=30,
        device="cpu",
    ):
        if messages is not None:
            return cls.tokenizer(
                messages,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)
        elif (m1 is not None) and (m2 is not None):
            return cls.tokenizer(
                m1,
                m2,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)
        else:
            raise ValueError("'messages' or 'm1' & 'm2' should be passed.")

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

    @classmethod
    def mean_pooling(cls, bert_lhs, attention_mask=None):
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

    @classmethod
    def _replace_cls_embedding(cls, encoder_lhs, attention_mask):
        encoder_output = cls.mean_pooling(
            bert_lhs=encoder_lhs,
            attention_mask=attention_mask,
        )
        encoder_output = torch.cat(
            tensors=(encoder_output.unsqueeze(1), encoder_lhs[:, 1:, :]),
            dim=1,
        )
        return encoder_output
