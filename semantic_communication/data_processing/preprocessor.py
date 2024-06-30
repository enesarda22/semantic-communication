import re
from typing import List

import nltk
from w3lib.html import replace_tags

from sklearn.model_selection import train_test_split

from semantic_communication.utils.general import RANDOM_STATE


class Preprocessor:
    encoder_fn = "encoder.pt"
    train_data_fn = "train_data.pt"
    val_data_fn = "val_data.pt"
    test_data_fn = "test_data.pt"
    next_sentence_pred_encoder_fn = "next_sentence_pred_encoder.pt"
    next_sentence_pred_train_data_fn = "next_sentence_pred_train_data.pt"
    next_sentence_pred_val_data_fn = "next_sentence_pred_val_data.pt"
    next_sentence_pred_test_data_fn = "next_sentence_pred_test_data.pt"

    @staticmethod
    def preprocess(m: str) -> (List[str], List[str]):
        # partition to sentences
        sentences = nltk.sent_tokenize(replace_tags(m, " "))

        def good_sentence(s):
            code_exists = re.search(r"\d{4}/", s) is not None
            break_exists = "\n" in s
            is_english = s.isascii()
            return (not code_exists) and (not break_exists) and is_english

        if len(sentences) % 2 != 0:
            sentences = sentences[:-1]

        first_sentences = []
        second_sentences = []

        for i in range(0, len(sentences) - 1, 2):
            if good_sentence(sentences[i]) and good_sentence(sentences[i + 1]):
                first_sentences.append(sentences[i])
                second_sentences.append(sentences[i + 1])

        return first_sentences, second_sentences

    @staticmethod
    def split_data(
        input_ids,
        attention_mask,
        train_size: float,
        test_size: float,
    ):
        (
            train_input_ids,
            temp_input_ids,
            train_attention_mask,
            temp_attention_mask,
        ) = train_test_split(
            input_ids,
            attention_mask,
            train_size=train_size,
            random_state=RANDOM_STATE,
        )

        (
            test_input_ids,
            val_input_ids,
            test_attention_mask,
            val_attention_mask,
        ) = train_test_split(
            temp_input_ids,
            temp_attention_mask,
            train_size=test_size,
            random_state=RANDOM_STATE,
        )

        return (
            train_input_ids,
            train_attention_mask,
            val_input_ids,
            val_attention_mask,
            test_input_ids,
            test_attention_mask,
        )
