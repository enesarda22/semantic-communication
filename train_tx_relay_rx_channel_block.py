import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.models.transceiver import (
    TxRelayRxChannelModel,
    Relay,
)
from semantic_communication.utils.channel import (
    init_channel,
    get_SNR,
)
from semantic_communication.models.semantic_decoder import SemanticDecoder
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

    parser.add_argument("--relay-decoder-path", type=str)
    add_semantic_decoder_args(parser)
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
    checkpoint = torch.load(args.relay_decoder_path, map_location=device)
    relay_decoder.load_state_dict(checkpoint["model_state_dict"])

    relay = Relay(
        semantic_encoder=semantic_encoder,
        semantic_decoder=relay_decoder,
        encoder=data_handler.encoder,
    )

    channel = init_channel(args.channel_type, args.sig_pow)
    tx_relay_rx_channel_model = TxRelayRxChannelModel(
        nin=args.channel_block_input_dim,
        n_latent=args.channel_block_latent_dim,
        channel=channel,
    ).to(device)

    optimizer = torch.optim.AdamW(
        params=tx_relay_rx_channel_model.parameters(),
        lr=args.lr,
    )
    criterion = torch.nn.MSELoss()

    for epoch in range(args.n_epochs):
        train_losses = []
        tx_relay_rx_channel_model.train()

        for b in tqdm(data_handler.train_dataloader):
            xb = b[0].to(device)
            attention_mask = b[1].to(device)

            encoder_output = semantic_encoder(
                input_ids=xb,
                attention_mask=attention_mask,
            )
            relay_out = relay(x=encoder_output[:, :-1, :])

            rel_SNR = get_SNR(args.SNR_min, args.SNR_max)
            tx_SNR = rel_SNR - args.SNR_diff
            output_hat = tx_relay_rx_channel_model(
                encoder_output[:, 1:, :], relay_out, tx_SNR, rel_SNR
            )
            ground_truth = torch.cat(
                [relay_out, encoder_output[:, 1:, :]], dim=-1
            )
            loss = criterion(output_hat, ground_truth)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_losses = []
        tx_relay_rx_channel_model.eval()
        for b in data_handler.val_dataloader:
            xb = b[0].to(device)
            attention_mask = b[1].to(device)

            encoder_output = semantic_encoder(
                input_ids=xb,
                attention_mask=attention_mask,
            )
            relay_out = relay(x=encoder_output[:, :-1, :])

            rel_SNR = get_SNR(args.SNR_min, args.SNR_max)
            tx_SNR = rel_SNR - args.SNR_diff
            with torch.no_grad():
                output_hat = tx_relay_rx_channel_model(
                    encoder_output[:, 1:, :], relay_out, tx_SNR, rel_SNR
                )

            ground_truth = torch.cat(
                [relay_out, encoder_output[:, 1:, :]], dim=-1
            )
            loss = criterion(output_hat, ground_truth)
            val_losses.append(loss.item())

        print("\n")
        print_loss(train_losses, "Train")
        print_loss(val_losses, "Val")

        mean_loss = np.mean(val_losses)

        checkpoint_path = os.path.join(
            args.checkpoint_path,
            f"tx-relay-rx-channel/tx_relay_rx_channel_{epoch}.pt",
        )

        create_checkpoint(
            path=checkpoint_path,
            model_state_dict=tx_relay_rx_channel_model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            mean_val_loss=mean_loss,
        )
