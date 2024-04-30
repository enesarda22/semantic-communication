import argparse

import torch
from tqdm import tqdm

from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.models.semantic_decoder import SemanticDecoder
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.models.semantic_transformer import SemanticTransformer
from semantic_communication.models.transceiver import (
    ChannelDecoder,
    ChannelEncoder,
    SrcRelayBlock,
)
from semantic_communication.utils.channel import init_channel, get_distance
from semantic_communication.utils.general import (
    add_semantic_decoder_args,
    add_data_args,
    add_channel_model_args,
    set_seed,
    get_device,
    load_model,
)


def generate_text():
    for b in tqdm(data_handler.test_dataloader):
        encoder_idx = b[0].to(device)
        encoder_attention_mask = b[1].to(device)

        encoder_idx = data_handler.label_encoder.transform(encoder_idx)

        d_sd = get_distance(args.d_min, args.d_max)
        d_sr = get_distance(d_sd * args.gamma_min, d_sd * args.gamma_max)

        predicted_ids, _ = src_relay_block.generate(
            input_ids=encoder_idx,
            attention_mask=encoder_attention_mask,
            d_sr=d_sr,
        )

        # find the end of sentences
        sep_indices = torch.argmax((predicted_ids == 2).long(), dim=1)
        input_ids_list = []
        for i in range(predicted_ids.shape[0]):
            k = sep_indices[i]
            if k == 0:  # no [SEP] predicted
                input_ids_list.append(predicted_ids[i, :])
            else:
                input_ids_list.append(predicted_ids[i, : k + 1])

        token_ids_list = [
            semantic_encoder.label_encoder.inverse_transform(input_ids)
            for input_ids in input_ids_list
        ]

        predicted_tokens = semantic_encoder.get_tokens(
            token_ids=token_ids_list,
            skip_special_tokens=True,
        )

        input_tokens = semantic_encoder.get_tokens(
            ids=encoder_idx,
            skip_special_tokens=True,
        )

        for i in range(len(input_tokens)):
            print(f"\nTrue Sentence: {input_tokens[i]}")
            for k in range(args.rate):
                print(f"Predicted Sentence {k}: {predicted_tokens[i * args.rate + k]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-relay-block-path", type=str)
    parser.add_argument("--batch-size", default=32, type=int)
    add_semantic_decoder_args(parser)
    add_data_args(parser)
    add_channel_model_args(parser)
    args = parser.parse_args()

    set_seed()
    device = get_device()

    data_handler = DataHandler(
        batch_size=args.batch_size,
        data_fp=args.data_fp,
    )

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

    semantic_transformer = SemanticTransformer(
        semantic_encoder=semantic_encoder,
        semantic_decoder=semantic_decoder,
    ).to(device)

    src_channel_encoder = ChannelEncoder(
        nin=args.channel_block_input_dim,
        nout=args.channel_block_latent_dim,
    ).to(device)

    relay_channel_decoder = ChannelDecoder(
        nin=args.channel_block_latent_dim,
        nout=args.channel_block_input_dim,
    ).to(device)

    channel = init_channel(args.channel_type, args.sig_pow, args.alpha, args.noise_pow)

    src_relay_block = SrcRelayBlock(
        semantic_transformer=semantic_transformer,
        src_channel_encoder=src_channel_encoder,
        relay_channel_decoder=relay_channel_decoder,
        channel=channel,
    ).to(device)
    load_model(src_relay_block, args.src_relay_block_path)

    src_relay_block.eval()
    generate_text()
