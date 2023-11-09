import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.models.transceiver import TxRelayChannelModel
from semantic_communication.utils.channel import AWGN, Rayleigh
from semantic_communication.utils.general import (
    get_device,
    print_loss,
    create_checkpoint,
    set_seed,
    add_channel_model_args,
    add_train_args,
    add_data_args,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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

    # Initializations
    SNR_dB = np.flip(np.arange(args.SNR_min, args.SNR_max + 1, args.SNR_step))

    if args.channel_type == "AWGN":
        channel = AWGN(SNR_dB[0], args.sig_pow)
    else:
        channel = Rayleigh(SNR_dB[0], args.sig_pow)

    tx_relay_channel_model = TxRelayChannelModel(
        nin=args.channel_block_input_dim,
        n_latent=args.channel_block_latent_dim,
        channel=channel,
    ).to(device)

    optimizer = torch.optim.AdamW(
        params=tx_relay_channel_model.parameters(),
        lr=args.lr,
    )
    criterion = torch.nn.MSELoss()

    cur_win, cur_SNR_index = 0, 0

    for epoch in range(args.n_epochs):
        train_losses = []
        tx_relay_channel_model.train()
        if cur_win >= args.SNR_window:
            cur_win = 0
            if not cur_SNR_index >= len(SNR_dB) - 1:
                cur_SNR_index += 1

            if args.channel_type == "AWGN":
                channel = AWGN(SNR_dB[cur_SNR_index], args.sig_pow)
            else:
                channel = Rayleigh(SNR_dB[cur_SNR_index], args.sig_pow)

            tx_relay_channel_model.channel = channel

        cur_win += 1
        for b in tqdm(data_handler.train_dataloader):
            xb = b[0].to(device)
            attention_mask = b[1].to(device)

            encoder_output = semantic_encoder(
                input_ids=xb,
                attention_mask=attention_mask,
            )
            encoder_output_hat = tx_relay_channel_model(encoder_output)
            loss = criterion(encoder_output_hat, encoder_output)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_losses = []
        tx_relay_channel_model.eval()
        for b in data_handler.val_dataloader:
            xb = b[0].to(device)
            attention_mask = b[1].to(device)

            encoder_output = semantic_encoder(
                input_ids=xb,
                attention_mask=attention_mask,
            )

            with torch.no_grad():
                encoder_output_hat = tx_relay_channel_model(encoder_output)

            loss = criterion(encoder_output_hat, encoder_output)
            val_losses.append(loss.item())

        print("\n")
        print_loss(train_losses, "Train")
        print_loss(val_losses, "Val")

        mean_loss = np.mean(val_losses)

        checkpoint_path = os.path.join(
            args.checkpoint_path,
            f"tx-relay-channel/tx_relay_channel_{epoch}.pt",
        )

        create_checkpoint(
            path=checkpoint_path,
            model_state_dict=tx_relay_channel_model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            mean_val_loss=mean_loss,
        )
