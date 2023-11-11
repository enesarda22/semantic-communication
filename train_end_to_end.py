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
)
from semantic_communication.utils.channel import (
    init_channel,
    get_SNR,
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
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # semantic decoders
    parser.add_argument("--relay-decoder-path", type=str)
    parser.add_argument("--receiver-decoder-path", type=str)
    add_semantic_decoder_args(parser)

    # channel models
    parser.add_argument("--tx-relay-channel-model-path", type=str)
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
    relay_checkpoint = torch.load(args.relay_decoder_path, map_location=device)
    relay_decoder.load_state_dict(relay_checkpoint["model_state_dict"])

    receiver_decoder = SemanticDecoder(
        vocab_size=data_handler.vocab_size,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        n_embeddings=args.n_embeddings * 2,
        block_size=args.max_length,
    ).to(device)
    rx_checkpoint = torch.load(args.receiver_decoder_path, map_location=device)
    receiver_decoder.load_state_dict(rx_checkpoint["model_state_dict"])

    channel = init_channel(args.channel_type, args.sig_pow)
    tx_relay_channel_model = TxRelayChannelModel(
        nin=args.channel_block_input_dim,
        n_latent=args.channel_block_latent_dim,
        channel=channel,
    ).to(device)

    if args.tx_relay_channel_model_path is not None:
        tx_relay_channel_model_checkpoint = torch.load(
            args.tx_relay_channel_model_path, map_location=device
        )
        tx_relay_channel_model.load_state_dict(
            tx_relay_channel_model_checkpoint["model_state_dict"]
        )

    tx_relay_rx_channel_model = TxRelayRxChannelModel(
        nin=args.channel_block_input_dim,
        n_latent=args.channel_block_latent_dim,
        channel=channel,
    ).to(device)

    if args.tx_relay_rx_channel_model_path is not None:
        tx_relay_rx_channel_model_checkpoint = torch.load(
            args.tx_relay_rx_channel_model_path, map_location=device
        )
        tx_relay_rx_channel_model.load_state_dict(
            tx_relay_rx_channel_model_checkpoint["model_state_dict"]
        )

    transceiver = Transceiver(
        semantic_encoder,
        relay_decoder,
        receiver_decoder,
        tx_relay_channel_model,
        tx_relay_rx_channel_model,
        data_handler.encoder,
    )
    optimizer = torch.optim.AdamW(transceiver.parameters(), lr=args.lr)

    best_loss = torch.inf
    for epoch in range(args.n_epochs):
        train_losses = []
        transceiver.train()
        for b in tqdm(data_handler.train_dataloader):
            xb = b[0].to(device)
            targets = data_handler.encode_token_ids(xb)
            attention_mask = b[1].to(device)

            rel_SNR = get_SNR(args.SNR_min, args.SNR_max)
            tx_SNR = rel_SNR - args.SNR_diff
            logits, loss = transceiver(
                xb, attention_mask, targets[:, 1:], tx_SNR, rel_SNR
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_losses = []
        transceiver.eval()
        for b in data_handler.val_dataloader:
            xb = b[0].to(device)
            targets = data_handler.encode_token_ids(xb)
            attention_mask = b[1].to(device)

            rel_SNR = get_SNR(args.SNR_min, args.SNR_max)
            tx_SNR = rel_SNR - args.SNR_diff
            with torch.no_grad():
                logits, loss = transceiver(
                    xb, attention_mask, targets[:, 1:], tx_SNR, rel_SNR
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
