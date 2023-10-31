from typing import List, Optional

import torch

from transformers import AutoTokenizer, AutoModel

from utils.general import get_device


class SemanticEncoder:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, max_length: int):
        self.device = get_device()
        self.max_length = max_length + 1

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.bert = AutoModel.from_pretrained(self.model_name).to(self.device)

    def __call__(
        self,
        messages: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        if messages is not None:
            input_ids = self.tokenize(messages=messages)

        self.bert.eval()
        with torch.no_grad():
            bert_out = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        bert_lhs = bert_out["last_hidden_state"]
        mean_pooling_out = self.mean_pooling(
            bert_lhs=bert_lhs,
            attention_mask=attention_mask,
        )
        out = torch.cat(
            tensors=(mean_pooling_out.unsqueeze(1), bert_lhs[:, 1:, :]),
            dim=1,
        )
        return out

    def tokenize(self, messages: List[str]):
        return self.tokenizer(
            messages,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

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
