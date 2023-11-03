import pickle
import re
from typing import List

import nltk
from w3lib.html import replace_tags


class Preprocessor:
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
