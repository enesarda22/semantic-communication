import argparse
from tqdm import tqdm

from semantic_communication.models.baseline_models import Tx_Relay, Tx_Relay_Rx
from semantic_communication.utils.general import (
    get_device,
    print_loss,
    create_checkpoint,
    set_seed,
    add_channel_model_args,
    add_train_args,
    add_data_args,
)
from semantic_communication.utils.channel import init_channel, get_SNR
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.data_processing.data_handler import DataHandler
import torch
import numpy as np
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-tx-relay-path", type=str)
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

    channel = init_channel(args.channel_type, args.sig_pow)

    num_classes = data_handler.vocab_size
    tx_relay_model = Tx_Relay(num_classes, args.channel_block_input_dim, args.channel_block_latent_dim, channel=channel, entire_network_train=1).to(device)
    checkpoint = torch.load(args.baseline_tx_relay_path)
    tx_relay_model.load_state_dict(checkpoint["model_state_dict"])

    tx_relay_rx_model = Tx_Relay_Rx(num_classes, args.channel_block_input_dim, args.channel_block_latent_dim, channel, tx_relay_model).to(device)

    optimizer = torch.optim.AdamW(
        params=tx_relay_rx_model.parameters(),
        lr=args.lr,
    )

    for epoch in range(args.n_epochs):
        train_losses = []
        tx_relay_rx_model.train()

        for b in tqdm(data_handler.train_dataloader):
            xb = b[0].to(device)
            attention_mask = b[1].to(device)

            xb = data_handler.encode_token_ids(xb)
            SNR = get_SNR(args.SNR_min, args.SNR_max)

            x_hat, loss = tx_relay_rx_model(xb[:, 1:], attention_mask[:, 1:], SNR, SNR - args.SNR_diff)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_losses = []
        tx_relay_rx_model.eval()
        for b in data_handler.val_dataloader:
            xb = b[0].to(device)
            attention_mask = b[1].to(device)

            xb = data_handler.encode_token_ids(xb)
            SNR = get_SNR(args.SNR_min, args.SNR_max)

            with torch.no_grad():
                x_hat, loss = tx_relay_rx_model(xb[:, 1:], attention_mask[:, 1:], SNR, SNR - args.SNR_diff)

            val_losses.append(loss.item())

        print("\n")
        print_loss(train_losses, "Train")
        print_loss(val_losses, "Val")

        mean_loss = np.mean(val_losses)
        create_checkpoint(
            path=os.path.join(
                args.checkpoint_path,
                f"baseline-tx-relay-rx/baseline_tx_relay_rx_{epoch}.pt",
            ),
            model_state_dict=tx_relay_rx_model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            mean_val_loss=mean_loss,
        )
        best_loss = mean_loss