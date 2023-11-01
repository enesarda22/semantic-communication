import argparse
from tqdm import tqdm

from baseline_models.tx_relay_rx_models import Tx_Relay, Tx_Relay_Rx
from utils.general import (
    get_device,
    print_loss,
    create_checkpoint,
)
from utils.channel import AWGN, Rayleigh
from data_processing.semantic_encoder import SemanticEncoder
from data_processing.data_handler import DataHandler
import torch
import numpy as np
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tx-relay-path", type=str)
    parser.add_argument("--checkpoint-path", default="checkpoints", type=str)
    parser.add_argument("--n-samples", default=10000, type=int)
    parser.add_argument("--train-size", default=0.9, type=float)
    parser.add_argument("--val-size", default=0.2, type=float)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--n-epochs", default=10, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--max-length", default=30, type=int)

    # New args
    parser.add_argument("--sig-pow", default=1.0, type=float)
    parser.add_argument("--SNR-diff", default=3.0, type=float)
    parser.add_argument("--SNR-min", default=3, type=int)
    parser.add_argument("--SNR-max", default=21, type=int)
    parser.add_argument("--SNR-step", default=3, type=int)
    parser.add_argument("--SNR-window", default=5, type=int)
    parser.add_argument("--channel-type", default="AWGN", type=str)
    args = parser.parse_args()

    device = get_device()

    if args.channel_type == "AWGN":
        tx_rx_channel = AWGN(0 - args.SNR_diff, args.sig_pow)
        tx_relay_channel = AWGN(0, args.sig_pow)
        relay_rx_channel = AWGN(0, args.sig_pow)

    else:
        tx_rx_channel = Rayleigh(0 - args.SNR_diff, args.sig_pow)
        tx_relay_channel = Rayleigh(0, args.sig_pow)
        relay_rx_channel = Rayleigh(0, args.sig_pow)

    semantic_encoder = SemanticEncoder(max_length=args.max_length)
    data_handler = DataHandler(
        semantic_encoder=semantic_encoder,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        train_size=args.train_size,
        val_size=args.val_size
    )

    data_handler.load_data()
    SNR_dB = np.flip(np.arange(args.SNR_min, args.SNR_max + 1, args.SNR_step))

    if args.channel_type == "AWGN":
        tx_rx_channel = AWGN(SNR_dB[0] - args.SNR_diff, args.sig_pow)
        tx_relay_channel = AWGN(SNR_dB[0], args.sig_pow)
        relay_rx_channel = AWGN(SNR_dB[0], args.sig_pow)

    else:
        tx_rx_channel = Rayleigh(SNR_dB[0] - args.SNR_diff, args.sig_pow)
        tx_relay_channel = Rayleigh(SNR_dB[0], args.sig_pow)
        relay_rx_channel = Rayleigh(SNR_dB[0], args.sig_pow)

    num_classes = data_handler.vocab_size
    tx_relay_model = Tx_Relay(num_classes, 384, 128, channel=tx_relay_channel, entire_network_train=1).to(device)
    checkpoint = torch.load(args.tx_relay_path)
    tx_relay_model.load_state_dict(checkpoint["model_state_dict"])

    tx_relay_rx_model = Tx_Relay_Rx(num_classes, 384, 128, tx_rx_channel, relay_rx_channel,tx_relay_model).to(device)
    optimizer = torch.optim.AdamW(
        params=tx_relay_rx_model.parameters(),
        lr=args.lr,
    )

    best_loss = 5
    cur_win, cur_SNR_index = 0, 0

    for epoch in range(args.n_epochs):
        train_losses = []
        tx_relay_rx_model.train()
        if cur_win >= args.SNR_window:
            cur_win = 0
            if not cur_SNR_index >= len(SNR_dB) - 1:
                cur_SNR_index += 1

            if args.channel_type == "AWGN":
                tx_rx_channel = AWGN(
                    SNR_dB[cur_SNR_index] - args.SNR_diff, args.sig_pow
                )
                tx_relay_channel = AWGN(
                    SNR_dB[cur_SNR_index], args.sig_pow
                )
                relay_rx_channel = AWGN(
                    SNR_dB[cur_SNR_index], args.sig_pow
                )

            else:
                tx_rx_channel = Rayleigh(
                    SNR_dB[cur_SNR_index] - args.SNR_diff, args.sig_pow
                )
                tx_relay_channel = Rayleigh(
                    SNR_dB[cur_SNR_index], args.sig_pow
                )
                relay_rx_channel = Rayleigh(
                    SNR_dB[cur_SNR_index], args.sig_pow
                )

            tx_relay_rx_model.tx_rx_channel = tx_rx_channel
            tx_relay_rx_model.relay_rx_channel = relay_rx_channel
            tx_relay_rx_model.tx_relay_model.channel = tx_relay_channel
        cur_win += 1

        for b in tqdm(data_handler.train_dataloader):
            xb = b[0].to(device)
            attention_mask = b[1].to(device)

            x_hat, loss = tx_relay_rx_model(xb[:, 1:], attention_mask[:, 1:])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_losses = []
        tx_relay_rx_model.eval()
        for b in data_handler.val_dataloader:
            xb = b[0].to(device)
            attention_mask = b[1].to(device)

            with torch.no_grad():
                x_hat, loss = tx_relay_rx_model(xb[:, 1:], attention_mask[:, 1:])

            val_losses.append(loss.item())

        print("\n")
        print_loss(train_losses, "Train")
        print_loss(val_losses, "Val")

        mean_loss = np.mean(val_losses)
        if mean_loss < best_loss:
            create_checkpoint(
                path=os.path.join(
                    args.checkpoint_path,
                    f"baseline-tx-relay-rx/baseline_tx_relay_rx_{epoch}.pt",
                ),
                model_state_dict=tx_relay_model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                mean_val_loss=mean_loss,
            )
            best_loss = mean_loss