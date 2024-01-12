import argparse

import numpy as np
import torch

from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.models.semantic_decoder import SemanticDecoder
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.utils.general import (
    get_device,
    set_seed,
    add_semantic_decoder_args,
    add_data_args,
    shift_inputs,
)


def generate_text():
    model.eval()
    xb, attention_mask = next(iter(data_handler.train_dataloader))
    xb = xb.to(device)
    attention_mask = attention_mask.to(device)
    B, T = xb.shape

    encoder_output = semantic_encoder(
        input_ids=xb,
        attention_mask=attention_mask,
    )
    xb_ids = data_handler.label_encoder.transform(xb)

    # find at which index the paddings start to occur for every sentence
    pad_occurrence = attention_mask.sum(dim=1)

    xb = torch.repeat_interleave(xb, pad_occurrence, dim=0)
    xb_attention_mask = torch.repeat_interleave(attention_mask, pad_occurrence, dim=0)

    # tril mask allows the prediction of each word in the sentence sequentially
    tril_mask = torch.cat([torch.tril(torch.ones(i, T)) for i in pad_occurrence], dim=0)
    xb_attention_mask = torch.minimum(xb_attention_mask, tril_mask.to(device))

    input_tokens = data_handler.get_tokens(
        token_ids=xb,
        attention_mask=xb_attention_mask,
        skip_special_tokens=True,
    )

    idx, encoder_output, attention_mask, targets = shift_inputs(
        xb_ids, encoder_output, attention_mask, args.mode
    )

    if args.mode == "sentence":
        predicted_ids = model.generate(
            encoder_output=encoder_output,
            max_length=args.max_length,
        )
        predicted_tokens = data_handler.get_tokens(
            ids=predicted_ids,
            skip_special_tokens=True,
        )
        input_tokens = data_handler.get_tokens(
            ids=idx,
            skip_special_tokens=True,
        )

        for input_, predicted in zip(input_tokens, predicted_tokens):
            print(f"{input_}\n{predicted}\n")

    else:
        predicted_ids = model.generate_next(
            idx=idx,
            encoder_output=encoder_output,
            attention_mask=attention_mask,
            sample=False,
        )
        # only keep non-padding ids
        predicted_ids = torch.masked_select(
            predicted_ids,
            attention_mask == 1,
        )
        predicted_tokens = data_handler.get_tokens(
            ids=predicted_ids.reshape(-1, 1),
            skip_special_tokens=True,
        )

        # find split indices based on number of paddings
        input_split_indices = pad_occurrence.cumsum(0)
        input_tokens = np.array_split(input_tokens, input_split_indices)

        pred_split_indices = attention_mask.sum(1).cumsum(0)
        predicted_tokens = np.array_split(predicted_tokens, pred_split_indices)

        for input_b, predicted_b in zip(input_tokens, predicted_tokens):
            for input_, predicted in zip(input_b, predicted_b):
                print(f"{input_} -> {predicted}")
            print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--relay-decoder-path", type=str)
    parser.add_argument("--batch-size", default=32, type=int)
    add_semantic_decoder_args(parser)
    add_data_args(parser)
    args = parser.parse_args()

    set_seed()
    device = get_device()

    semantic_encoder = SemanticEncoder(max_length=args.max_length)
    data_handler = DataHandler(
        semantic_encoder=semantic_encoder,
        data_fp=args.data_fp,
        batch_size=args.batch_size,
    )

    model = SemanticDecoder(
        vocab_size=data_handler.vocab_size,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        n_embeddings=args.n_embeddings,
        block_size=args.max_length,
        semantic_encoder=semantic_encoder,
        label_encoder=data_handler.label_encoder,
    ).to(device)
    checkpoint = torch.load(args.relay_decoder_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    with torch.no_grad():
        generate_text()
