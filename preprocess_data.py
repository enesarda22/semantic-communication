import argparse
import glob
import os
from transformers import AutoTokenizer

from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset

from semantic_communication.data_processing.preprocessor import Preprocessor
from semantic_communication.utils.tensor_label_encoder import TensorLabelEncoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--europarl-folder-path", default="europarl/txt", type=str)
    parser.add_argument("--output-data-fp", default="", type=str)
    parser.add_argument("--max-length", default=30, type=int)
    parser.add_argument("--train-size", default=0.7, type=float)
    parser.add_argument("--test-size", default=0.5, type=float)
    parser.add_argument("--n-samples", default=5000000, type=int)
    args = parser.parse_args()

    en_fp = os.path.join(args.europarl_folder_path, "en/*.txt")
    txt_filepaths = sorted(glob.glob(en_fp))

    preprocessed_messages_firsts = []
    preprocessed_messages_seconds = []

    for txt_fp in tqdm(txt_filepaths, "Txt files"):
        with open(txt_fp, "r", encoding="utf-8") as f:
            m = f.read()

        first_messages, second_messages = Preprocessor.preprocess(m)

        preprocessed_messages_firsts.extend(first_messages)
        preprocessed_messages_seconds.extend(second_messages)

        if args.n_samples and (args.n_samples < len(preprocessed_messages_firsts)):
            preprocessed_messages_firsts = preprocessed_messages_firsts[
                : args.n_samples
            ]
            preprocessed_messages_seconds = preprocessed_messages_seconds[
                : args.n_samples
            ]

            break

    print(f"Number of sentences: {len(preprocessed_messages_seconds)}")

    # tokenize
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    first_tokens = tokenizer(
        preprocessed_messages_firsts,
        padding="max_length",
        max_length=args.max_length + 1,
        truncation=True,
        return_tensors="pt",
    )

    second_tokens = tokenizer(
        preprocessed_messages_seconds,
        padding="max_length",
        max_length=args.max_length + 1,
        truncation=True,
        return_tensors="pt",
    )

    # drop sentences shorter than 2 tokens for firsts
    long_sentence_query_firsts = first_tokens["attention_mask"].sum(dim=1) > 4
    long_sentence_query_seconds = second_tokens["attention_mask"].sum(dim=1) > 4
    long_sentence_query = long_sentence_query_firsts & long_sentence_query_seconds

    first_input_ids = first_tokens["input_ids"][long_sentence_query, :]
    first_attention_mask = first_tokens["attention_mask"][long_sentence_query, :]

    second_input_ids = second_tokens["input_ids"][long_sentence_query, :]
    second_attention_mask = second_tokens["attention_mask"][long_sentence_query, :]

    combined_input_ids = torch.cat((first_input_ids, second_input_ids[:, 1:]), dim=1)
    combined_attention_masks = torch.cat(
        (first_attention_mask, second_attention_mask[:, 1:]), dim=1
    )

    (
        train_input_ids,
        train_attention_mask,
        val_input_ids,
        val_attention_mask,
        test_input_ids,
        test_attention_mask,
    ) = Preprocessor.split_data(
        input_ids=combined_input_ids,
        attention_mask=combined_attention_masks,
        train_size=args.train_size,
        test_size=args.test_size,
    )

    # drop the unknown tokens in val/test sets
    unique_ids = torch.unique(train_input_ids)

    val_query = torch.all(torch.isin(val_input_ids, unique_ids), dim=1)
    val_input_ids = val_input_ids[val_query, :]
    val_attention_mask = val_attention_mask[val_query, :]

    test_query = torch.all(torch.isin(test_input_ids, unique_ids), dim=1)
    test_input_ids = test_input_ids[test_query, :]
    test_attention_mask = test_attention_mask[test_query, :]

    # train label encoder
    encoder_fp = os.path.join(
        args.output_data_fp, Preprocessor.next_sentence_pred_encoder_fn
    )
    encoder = TensorLabelEncoder().fit(train_input_ids)
    torch.save(encoder, encoder_fp)

    train_dataset = TensorDataset(train_input_ids, train_attention_mask)
    val_dataset = TensorDataset(val_input_ids, val_attention_mask)
    test_dataset = TensorDataset(test_input_ids, test_attention_mask)

    train_fp = os.path.join(
        args.output_data_fp, Preprocessor.next_sentence_pred_train_data_fn
    )
    torch.save(train_dataset, train_fp)

    val_fp = os.path.join(
        args.output_data_fp, Preprocessor.next_sentence_pred_val_data_fn
    )
    torch.save(val_dataset, val_fp)

    test_fp = os.path.join(
        args.output_data_fp, Preprocessor.next_sentence_pred_test_data_fn
    )
    torch.save(test_dataset, test_fp)
