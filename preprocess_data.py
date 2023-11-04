import argparse
import glob
import os

from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset
from sklearn.preprocessing import LabelEncoder

from semantic_communication.data_processing.preprocessor import Preprocessor
from semantic_communication.models.semantic_encoder import SemanticEncoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--europarl-folder-path", type=str)
    parser.add_argument("--output-data-fp", default="", type=str)
    parser.add_argument("--max-length", default=30, type=int)
    parser.add_argument("--train-size", default=0.7, type=float)
    parser.add_argument("--test-size", default=0.5, type=float)
    parser.add_argument("--n-samples", default=None, type=int)
    args = parser.parse_args()

    en_fp = os.path.join(args.europarl_folder_path, "en/*.txt")
    txt_filepaths = sorted(glob.glob(en_fp))

    preprocessed_messages = []
    for txt_fp in tqdm(txt_filepaths, "Txt files"):
        with open(txt_fp, "r", encoding="utf-8") as f:
            m = f.read()

        preprocessed_messages.extend(Preprocessor.preprocess(m))
        if args.n_samples and (args.n_samples < len(preprocessed_messages)):
            preprocessed_messages = preprocessed_messages[: args.n_samples]
            break

    print(f"Number of sentences: {len(preprocessed_messages)}")

    # tokenize
    semantic_encoder = SemanticEncoder(max_length=args.max_length)
    tokens = semantic_encoder.tokenize(messages=preprocessed_messages)

    # train label encoder
    encoder_fp = os.path.join(args.output_data_fp, Preprocessor.encoder_fn)
    encoder = LabelEncoder().fit(tokens["input_ids"].flatten().to("cpu"))
    torch.save(encoder, encoder_fp)

    (
        train_input_ids,
        train_attention_mask,
        val_input_ids,
        val_attention_mask,
        test_input_ids,
        test_attention_mask,
    ) = Preprocessor.split_data(
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
        train_size=args.train_size,
        test_size=args.test_size,
    )

    train_dataset = TensorDataset(train_input_ids, train_attention_mask)
    val_dataset = TensorDataset(val_input_ids, val_attention_mask)
    test_dataset = TensorDataset(test_input_ids, test_attention_mask)

    train_fp = os.path.join(args.output_data_fp, Preprocessor.train_data_fn)
    torch.save(train_dataset, train_fp)

    val_fp = os.path.join(args.output_data_fp, Preprocessor.val_data_fn)
    torch.save(val_dataset, val_fp)

    test_fp = os.path.join(args.output_data_fp, Preprocessor.test_data_fn)
    torch.save(test_dataset, test_fp)
