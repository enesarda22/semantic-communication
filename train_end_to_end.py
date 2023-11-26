import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.models.transceiver import (
    TxRelayChannelModel,
    TxRelayRxChannelModel,
    Transceiver,
    RelayChannelBlock,
    ChannelEncoder,
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
    parser.add_argument("--receiver-decoder-path", type=str)
    add_semantic_decoder_args(parser)

    # channel models
    parser.add_argument("--relay-channel-block-path", type=str)
    parser.add_argument("--tx-relay-rx-channel-model-path", type=str)
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

    tx_channel_enc = ChannelEncoder(
        nin=args.channel_block_input_dim,
        nout=args.channel_block_latent_dim,
    ).to(device)

    channel = init_channel(args.channel_type, args.sig_pow, args.alpha, args.noise_pow)
    tx_relay_channel_model = TxRelayChannelModel(
        nin=args.channel_block_input_dim,
        n_latent=args.channel_block_latent_dim,
        channel=channel,
    ).to(device)

    relay_channel_block = RelayChannelBlock(
        semantic_decoder=relay_decoder,
        tx_channel_enc=tx_channel_enc,
        tx_relay_channel_enc_dec=tx_relay_channel_model,
    ).to(device)
    load_model(relay_channel_block, args.relay_channel_block_path)

    # freeze
    for param in relay_channel_block.parameters():
        param.requires_grad = False

    receiver_decoder = SemanticDecoder(
        vocab_size=data_handler.vocab_size,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        n_embeddings=args.n_embeddings * 2,
        block_size=args.max_length,
    ).to(device)
    load_model(receiver_decoder, args.receiver_decoder_path)

    tx_relay_rx_channel_model = TxRelayRxChannelModel(
        nin=args.channel_block_input_dim,
        n_latent=args.channel_block_latent_dim,
        channel=channel,
    ).to(device)
    load_model(tx_relay_rx_channel_model, args.tx_relay_rx_channel_model_path)

    transceiver = Transceiver(
        semantic_encoder=semantic_encoder,
        relay_channel_block=relay_channel_block,
        rx_semantic_decoder=receiver_decoder,
        tx_relay_rx_channel_enc_dec=tx_relay_rx_channel_model,
        encoder=data_handler.encoder,
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
            targets = data_handler.encode_token_ids(xb)
            attention_mask = b[1].to(device)

            d_sd = get_distance(args.d_min, args.d_max)
            d_sr = get_distance(d_sd * args.gamma_min, d_sd * args.gamma_max)
            d_rd = d_sd - d_sr

            logits, loss = transceiver(
                xb, attention_mask, targets[:, 1:], d_sd, d_sr, d_rd
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        val_losses = []
        transceiver.eval()
        for b in data_handler.val_dataloader:
            xb = b[0].to(device)
            targets = data_handler.encode_token_ids(xb)
            attention_mask = b[1].to(device)

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
