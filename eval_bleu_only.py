import argparse
import os

import numpy as np
import torch
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

from semantic_communication.models.semantic_transformer import SemanticTransformer
from semantic_communication.models.transceiver import (
    ChannelEncoder,
    ChannelDecoder,
    init_src_relay_transformer_from_transceiver,
)

from semantic_communication.utils.general import (
    get_device,
    set_seed,
    add_semantic_decoder_args,
    add_channel_model_args,
    add_data_args,
    load_model,
)

from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.models.semantic_decoder import SemanticDecoder
from semantic_communication.utils.channel import init_channel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model paths (choose one)
    parser.add_argument("--semantic-transformer-path", type=str, default=None)
    parser.add_argument("--transceiver-path", type=str, default=None)

    add_semantic_decoder_args(parser)
    add_channel_model_args(parser)
    add_data_args(parser)

    # test args
    parser.add_argument("--batch-size", default=125, type=int)
    parser.add_argument("--d-list", nargs="+", type=float, required=True)
    parser.add_argument("--n-test", default=500, type=int)

    args = parser.parse_args()
    device = get_device()
    set_seed()

    results_dir = "Results"
    os.makedirs(results_dir, exist_ok=True)

    data_handler = DataHandler(
        batch_size=args.batch_size,
        data_fp=args.data_fp,
        mode=args.mode,
    )

    # -------- Build SemanticTransformer only --------
    semantic_encoder = SemanticEncoder(
        label_encoder=data_handler.label_encoder,
        max_length=args.max_length,
        mode=args.mode,
        rate=args.rate,
    ).to(device)

    semantic_decoder = SemanticDecoder(
        vocab_size=data_handler.vocab_size,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        n_embeddings=args.n_embeddings,
        block_size=args.max_length,
        bert=semantic_encoder.bert,
        pad_idx=data_handler.label_encoder.pad_id,
    ).to(device)

    channel_encoder = ChannelEncoder(
        nin=args.channel_block_input_dim,
        nout=args.channel_block_latent_dim,
    ).to(device)

    channel_decoder = ChannelDecoder(
        nin=args.channel_block_latent_dim,
        nout=args.channel_block_input_dim,
    ).to(device)

    channel = init_channel(args.channel_type, args.sig_pow, args.alpha, args.noise_pow)

    semantic_transformer = SemanticTransformer(
        semantic_encoder=semantic_encoder,
        semantic_decoder=semantic_decoder,
        channel_encoder=channel_encoder,
        channel_decoder=channel_decoder,
        channel=channel,
    ).to(device)

    # Load weights:
    # - If semantic-transformer-path is given, load directly
    # - Else if transceiver-path is given, extract src/relay transformer weights
    if args.semantic_transformer_path is not None:
        load_model(semantic_transformer, args.semantic_transformer_path)
    elif args.transceiver_path is not None:
        state_dict = init_src_relay_transformer_from_transceiver(args.transceiver_path)
        semantic_transformer.load_state_dict(state_dict)
    else:
        raise ValueError("Provide --semantic-transformer-path or --transceiver-path")

    semantic_transformer.eval()

    # -------- BLEU-only evaluation --------
    n_d = len(args.d_list)
    mean_bleu = np.zeros((n_d, 1))
    std_bleu = np.zeros((n_d, 1))

    smoothing_function = SmoothingFunction().method1

    # Keep a single table across all distances (includes d in the sheet)
    records = []

    for distance_index, d in enumerate(args.d_list):
        print(f"Simulating for distance: {d}")

        bleu_scores = []

        for b in data_handler.test_dataloader:
            encoder_idx = b[0].to(device)
            encoder_attention_mask = b[1].to(device)

            # match your existing pipeline
            encoder_idx = data_handler.label_encoder.transform(encoder_idx)

            # SemanticTransformer.generate returns predicted_ids
            predicted_ids = semantic_transformer.generate(
                input_ids=encoder_idx,
                attention_mask=encoder_attention_mask,
                d=d,
            )

            # find end-of-sentence ([SEP]==2)
            sep_indices = torch.argmax((predicted_ids == 2).long(), dim=1)
            input_ids_list = []
            for i in range(predicted_ids.shape[0]):
                k = sep_indices[i]
                if k == 0:  # no [SEP] predicted
                    input_ids_list.append(predicted_ids[i, :])
                else:
                    input_ids_list.append(predicted_ids[i, : k + 1])

            token_ids_list = [
                semantic_encoder.label_encoder.inverse_transform(ids_)
                for ids_ in input_ids_list
            ]

            predicted_tokens = semantic_encoder.get_tokens(
                token_ids=token_ids_list,
                skip_special_tokens=True,
            )

            # Keep Code 2 behavior for next_sentence; otherwise match Code 1 style
            if args.mode == "next_sentence":
                input_tokens = semantic_encoder.get_tokens(
                    ids=encoder_idx[:, -args.max_length:],
                    skip_special_tokens=True,
                )
            else:
                input_tokens = semantic_encoder.get_tokens(
                    ids=encoder_idx,
                    skip_special_tokens=True,
                )

            for s1, s2 in zip(input_tokens, predicted_tokens):
                ref = word_tokenize(s1)
                hyp = word_tokenize(s2)

                bleu_score = sentence_bleu(
                    [ref],
                    hyp,
                    smoothing_function=smoothing_function,
                )

                bleu_scores.append(bleu_score)
                records.append([d, s1, s2, bleu_score])

            if len(bleu_scores) >= args.n_test:
                break

        n_test_samples = len(bleu_scores)
        mean_bleu[distance_index, 0] = float(np.mean(bleu_scores)) if n_test_samples else 0.0
        std_bleu[distance_index, 0] = (
            float(np.std(bleu_scores, ddof=1) / np.sqrt(n_test_samples))
            if n_test_samples > 1
            else 0.0
        )

        # Code 1–style filenames
        mean_fp = os.path.join(
            results_dir,
            f"{args.mode}_{args.channel_block_latent_dim}_{args.channel_type}_proposed_mean_bleu.npy",
        )
        std_fp = os.path.join(
            results_dir,
            f"{args.mode}_{args.channel_block_latent_dim}_{args.channel_type}_proposed_std_bleu.npy",
        )
        np.save(mean_fp, mean_bleu)
        np.save(std_fp, std_bleu)

        # Excel (BLEU only + sentence pairs; includes distance so you can filter)
        df = pd.DataFrame(
            records,
            columns=["d", "Sentence 1", "Sentence 2", "BLEU Score"],
        )
        xlsx_fp = os.path.join(
            results_dir,
            f"{args.mode}_{args.channel_block_latent_dim}_{args.channel_type}_proposed_output.xlsx",
        )
        df.to_excel(xlsx_fp, index=False)
