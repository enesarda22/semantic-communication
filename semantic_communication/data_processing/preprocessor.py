import pickle
import re
from typing import List

import nltk
from sklearn.model_selection import train_test_split
from w3lib.html import replace_tags

from semantic_communication.utils.general import RANDOM_STATE


class Preprocessor:
    train_data_fn = "train_data.pkl"
    val_data_fn = "val_data.pkl"
    test_data_fn = "test_data.pkl"

    @staticmethod
    def dump_data(data, fp):
        with open(fp, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def preprocess(m: str) -> List[str]:
        # partition to sentences
        sentences = nltk.sent_tokenize(replace_tags(m, " "))

        def good_sentence(s):
            code_exists = re.search(r"\d{4}/", s) is not None
            break_exists = "\n" in s
            is_english = s.isascii()
            return (not code_exists) and (not break_exists) and is_english

        # filter sentences
        sentences = [s.lower() for s in sentences if good_sentence(s)]
        return sentences

    @staticmethod
    def split_data(messages: List[str], train_size: float, test_size: float):
        train_messages, temp_messages = train_test_split(
            messages,
            train_size=train_size,
            random_state=RANDOM_STATE,
        )

        test_messages, val_messages = train_test_split(
            temp_messages,
            train_size=test_size,
            random_state=RANDOM_STATE,
        )

        return train_messages, val_messages, test_messages
