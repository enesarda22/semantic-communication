import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.models.paraphrase_detector import ParaphraseDetector
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.utils.general import (
    add_data_args,
    add_paraphrase_detector_args,
    get_device,
    set_seed,
    load_model,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paraphrase-detector-path", default=None, type=str)
    parser.add_argument("--batch-size", default=32, type=int)
    add_paraphrase_detector_args(parser)
    add_data_args(parser)
    args = parser.parse_args()

    set_seed()
    device = get_device()

    data_handler = DataHandler(
        batch_size=args.batch_size,
        data_fp=args.data_fp,
        fn_prefix="paraphrase_",
    )

    semantic_encoder = SemanticEncoder(
        label_encoder=data_handler.label_encoder,
        max_length=args.max_length * 2,
        mode="sentence",
    ).to(device)

    paraphrase_detector = ParaphraseDetector(
        semantic_encoder=semantic_encoder,
        n_in=args.n_in,
        n_latent=args.n_latent,
    ).to(device)
    load_model(paraphrase_detector, args.paraphrase_detector_path)

    labels_arr = np.empty(0)
    probs_arr = np.empty(0)
    for b in tqdm(data_handler.test_dataloader):
        encoder_idx = b[0].to(device)
        encoder_attention_mask = b[1].to(device)
        labels = b[2].to(device)

        encoder_idx = data_handler.label_encoder.transform(encoder_idx)
        with torch.no_grad():
            logits = paraphrase_detector(
                input_ids=encoder_idx,
                attention_mask=encoder_attention_mask,
            )
            probs = F.sigmoid(logits)

        labels_arr = np.concatenate([labels_arr, labels.detach().cpu().numpy()])
        probs_arr = np.concatenate([probs_arr, probs.detach().cpu().numpy()])

        split_idx = torch.argmax((encoder_idx == 2).int(), dim=1)
        for i in range(encoder_idx.shape[0]):
            s1 = semantic_encoder.get_tokens(
                ids=encoder_idx[[i], : split_idx[i] + 1],
                skip_special_tokens=True,
            )[0]
            s2 = semantic_encoder.get_tokens(
                ids=encoder_idx[[i], split_idx[i] + 1 :],
                skip_special_tokens=True,
            )[0]
            print(s1)
            print(s2)
            print(f"Paraphrase likelihood: {probs[i]:.4f}")
            print(f"Label: {labels[i]:.0f}\n")

    acc = np.mean(labels_arr.astype(bool) == (probs_arr > 0.5))
    print(f"Accuracy: {acc:.4f}")
