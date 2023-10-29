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
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", default="checkpoints", type=str)
    parser.add_argument("--n-samples", default=10000, type=int)
    parser.add_argument("--train-size", default=0.9, type=float)
    parser.add_argument("--val-size", default=0.2, type=float)
    parser.add_argument("--max-length", default=30, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--n-epochs", default=10, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--n-heads", default=4, type=int)
    parser.add_argument("--n-embeddings", default=384, type=int)

    # New args
    parser.add_argument("--sig-pow", default=1.0, type=float)
    parser.add_argument("--SNR-min", default=3, type=int)
    parser.add_argument("--SNR-max", default=24, type=int)
    parser.add_argument("--SNR-step", default=3, type=int)
    parser.add_argument("--SNR-window", default=5, type=int)
    parser.add_argument("--channel-type", default="AWGN", type=str)
    args = parser.parse_args()

    device = get_device()

    semantic_encoder = SemanticEncoder(max_length=args.max_length)
    data_handler = DataHandler(
        semantic_encoder=semantic_encoder,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        train_size=args.train_size,
        val_size=args.val_size
    )
    data_handler.load_data()

    # Initializations
    SNR_dB = np.flip(np.arange(args.SNR_min, args.SNR_max, args.SNR_step))

    if args.channel_type == "AWGN":
        channel = AWGN(SNR_dB[0], args.sig_pow)
    else:
        channel = Rayleigh(SNR_dB[0], args.sig_pow)

    tx_relay_channel_model = TxRelayChannelModel(384, 128, channel).to(device)

    optimizer = torch.optim.AdamW(
        params=tx_relay_channel_model.parameters(),
        lr=args.lr,
    )
    criterion = torch.nn.MSELoss()

    best_loss = 5
    cur_win, cur_SNR_index = 0, 0

    for epoch in range(args.n_epochs):
        train_losses = []
        tx_relay_channel_model.train()
        for b in tqdm(data_handler.train_dataloader):
            if cur_win > args.SNR_window:
                cur_win = 0
                if not cur_SNR_index >= len(SNR_dB) - 1:
                    cur_SNR_index += 1

                if args.channel_type == "AWGN":
                    channel = AWGN(SNR_dB[cur_SNR_index], args.sig_pow)
                else:
                    channel = Rayleigh(SNR_dB[cur_SNR_index], args.sig_pow)

                tx_relay_channel_model.channel = channel

            cur_win += 1

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

        if mean_loss < best_loss:
            create_checkpoint(
                path=os.path.join(
                    args.checkpoint_path,
                    f"tx-relay-channel/tx_relay_channel_{epoch}.pt",
                ),
                model_state_dict=tx_relay_channel_model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                mean_val_loss=mean_loss,
            )
            best_loss = mean_loss
