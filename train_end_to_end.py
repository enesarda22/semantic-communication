import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.models.transceiver import (
    Transceiver,
    RelayChannelBlock,
    ChannelEncoder,
    SrcRelayChannelModel,
    SrcRelayDstChannelModel,
)
from semantic_communication.utils.channel import (
    init_channel,
    get_distance,
)
from semantic_communication.models.semantic_decoder import SemanticDecoder
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.utils.general import (
    get_device,
    print_loss,
    create_checkpoint,
    set_seed,
    add_semantic_decoder_args,
    add_channel_model_args,
    add_data_args,
    add_train_args,
    load_model,
    load_optimizer,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transceiver-path", type=str)

    # semantic decoders
    parser.add_argument("--dst-decoder-path", type=str)
    add_semantic_decoder_args(parser)

    # channel models
    parser.add_argument("--relay-channel-block-path", type=str)
    parser.add_argument("--src-relay-dst-channel-model-path", type=str)
    add_channel_model_args(parser)

    add_data_args(parser)
    add_train_args(parser)
    args = parser.parse_args()

    set_seed()
    device = get_device()

    semantic_encoder = SemanticEncoder(max_length=args.max_length)
    data_handler = DataHandler(
        semantic_encoder=semantic_encoder,
        batch_size=args.batch_size,
        data_fp=args.data_fp,
    )

    relay_decoder = SemanticDecoder(
        vocab_size=data_handler.vocab_size,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        n_embeddings=args.n_embeddings,
        block_size=args.max_length,
    ).to(device)

    src_channel_enc = ChannelEncoder(
        nin=args.channel_block_input_dim,
        nout=args.channel_block_latent_dim,
    ).to(device)

    channel = init_channel(args.channel_type, args.sig_pow, args.alpha, args.noise_pow)
    src_relay_channel_model = SrcRelayChannelModel(
        n_in=args.channel_block_input_dim,
        n_latent=args.channel_block_latent_dim,
        channel=channel,
    ).to(device)

    relay_channel_block = RelayChannelBlock(
        source_channel_encoder=src_channel_enc,
        src_relay_channel_model=src_relay_channel_model,
        semantic_decoder=relay_decoder,
    ).to(device)
    load_model(relay_channel_block, args.relay_channel_block_path)

    # freeze
    for param in relay_channel_block.parameters():
        param.requires_grad = False

    dst_decoder = SemanticDecoder(
        vocab_size=data_handler.vocab_size,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        n_embeddings=args.n_embeddings * 2,
        block_size=args.max_length,
    ).to(device)
    load_model(dst_decoder, args.dst_decoder_path)

    src_relay_dst_channel_model = SrcRelayDstChannelModel(
        n_in=args.channel_block_input_dim,
        n_latent=args.channel_block_latent_dim,
        channel=channel,
    ).to(device)
    load_model(src_relay_dst_channel_model, args.src_relay_dst_channel_model_path)

    transceiver = Transceiver(
        semantic_encoder=semantic_encoder,
        relay_channel_block=relay_channel_block,
        dst_semantic_decoder=dst_decoder,
        src_relay_dst_channel_model=src_relay_dst_channel_model,
        label_encoder=data_handler.label_encoder,
    )
    load_model(transceiver, args.transceiver_path)

    optimizer = torch.optim.AdamW(transceiver.parameters(), lr=args.lr)
    load_optimizer(optimizer, args.transceiver_path)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=args.lr,
        total_steps=args.n_epochs,
    )

    best_loss = torch.inf
    for epoch in range(args.n_epochs):
        train_losses = []
        transceiver.train()
        for b in tqdm(data_handler.train_dataloader):
            xb = b[0].to(device)
            attention_mask = b[1].to(device)
            targets = data_handler.label_encoder.transform(xb)

            d_sd = get_distance(args.d_min, args.d_max)
            d_sr = get_distance(d_sd * args.gamma_min, d_sd * args.gamma_max)
            d_rd = d_sd - d_sr

            _, loss = transceiver(xb, attention_mask, targets[:, 1:], d_sd, d_sr, d_rd)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        val_losses = []
        transceiver.eval()
        for b in data_handler.val_dataloader:
            xb = b[0].to(device)
            attention_mask = b[1].to(device)
            targets = data_handler.label_encoder.transform(xb)

            d_sd = get_distance(args.d_min, args.d_max)
            d_sr = get_distance(d_sd * args.gamma_min, d_sd * args.gamma_max)
            d_rd = d_sd - d_sr
            with torch.no_grad():
                _, loss = transceiver(
                    xb, attention_mask, targets[:, 1:], d_sd, d_sr, d_rd
                )

            val_losses.append(loss.item())

        print("\n")
        print_loss(train_losses, "Train")
        print_loss(val_losses, "Val")

        mean_loss = np.mean(val_losses)

        checkpoint_path = os.path.join(
            args.checkpoint_path,
            f"end-to-end-transceiver/end_to_end_transceiver_{epoch}.pt",
        )

        if mean_loss < best_loss:
            create_checkpoint(
                path=checkpoint_path,
                model_state_dict=transceiver.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                mean_val_loss=mean_loss,
            )
            best_loss = mean_loss
        else:
            create_checkpoint(
                path=checkpoint_path,
                model_state_dict=None,
                optimizer_state_dict=None,
                mean_val_loss=mean_loss,
            )
