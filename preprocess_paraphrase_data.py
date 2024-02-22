import argparse
import csv
import os

import torch
from torch.utils.data import TensorDataset

from semantic_communication.data_processing.preprocessor import Preprocessor
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.utils.general import get_device


def get_tokens(fn, paws_qqp_fn=None):
    data = []
    with open(os.path.join(args.qqp_folder_path, fn)) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            data.append([row[1], row[2], row[0]])

    with open(os.path.join(args.paws_wiki_folder_path, fn)) as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip the header line
        for row in reader:
            data.append([row[-3], row[-2], row[-1]])

    if paws_qqp_fn is not None:
        with open(os.path.join(args.paws_qqp_folder_path, paws_qqp_fn)) as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)  # skip the header line
            for row in reader:
                data.append([row[-3], row[-2], row[-1]])

    m1 = [sample[0] for sample in data]
    m2 = [sample[1] for sample in data]
    tokens = SemanticEncoder.tokenize(
        m1=m1,
        m2=m2,
        max_length=args.max_length * 2,
        device=device,
    )
    labels = torch.FloatTensor([float(sample[2]) for sample in data])

    # filter the messages with unknown tokens
    q = torch.all(torch.isin(tokens["input_ids"], label_encoder.classes), dim=1)
    return tokens["input_ids"][q, :], tokens["attention_mask"][q, :], labels[q]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qqp-folder-path", type=str)
    parser.add_argument("--paws-qqp-folder-path", type=str)
    parser.add_argument("--paws-wiki-folder-path", type=str)
    parser.add_argument("--output-data-fp", default="", type=str)
    parser.add_argument("--max-length", default=30, type=int)
    args = parser.parse_args()

    device = get_device()
    label_encoder = torch.load(
        os.path.join(args.output_data_fp, Preprocessor.encoder_fn)
    )

    fn_prefix = "paraphrase_"

    train_input_ids, train_mask, train_labels = get_tokens("train.tsv", "train.tsv")
    train_dataset = TensorDataset(train_input_ids, train_mask, train_labels)
    train_fp = os.path.join(args.output_data_fp, fn_prefix + Preprocessor.train_data_fn)
    torch.save(train_dataset, train_fp)

    val_input_ids, val_mask, val_labels = get_tokens("dev.tsv", "dev_and_test.tsv")
    val_dataset = TensorDataset(val_input_ids, val_mask, val_labels)
    val_fp = os.path.join(args.output_data_fp, fn_prefix + Preprocessor.val_data_fn)
    torch.save(val_dataset, val_fp)

    test_input_ids, test_mask, test_labels = get_tokens("test.tsv")
    test_dataset = TensorDataset(test_input_ids, test_mask, test_labels)
    test_fp = os.path.join(args.output_data_fp, fn_prefix + Preprocessor.test_data_fn)
    torch.save(test_dataset, test_fp)
