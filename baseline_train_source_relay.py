import argparse
from tqdm import tqdm

from semantic_communication.models.baseline_models import Tx_Relay
from semantic_communication.utils.channel import (
    init_channel,
    get_distance,
)
from semantic_communication.data_processing.data_handler import DataHandler
import torch
import numpy as np
import os
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

    data_handler = DataHandler(
        batch_size=args.batch_size,
        data_fp=args.data_fp,
    )

    channel = init_channel(args.channel_type, args.sig_pow, args.alpha, args.noise_pow)
    num_classes = data_handler.vocab_size
    tx_relay_model = Tx_Relay(
        nin=num_classes,
        n_emb=args.channel_block_input_dim,
        n_latent=args.channel_block_latent_dim,
        channel=channel,
    ).to(device)

    optimizer = torch.optim.AdamW(
        params=tx_relay_model.parameters(),
        lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=args.lr,
        total_steps=args.n_epochs,
    )

    best_loss = torch.inf

    for epoch in range(args.n_epochs):
        train_losses = []
        tx_relay_model.train()
        for b in tqdm(data_handler.train_dataloader):
            encoder_idx = b[0].to(device)
            encoder_attention_mask = b[1].to(device)

            encoder_idx = data_handler.label_encoder.transform(encoder_idx)

            d_sd = get_distance(args.d_min, args.d_max)
            d_sr = get_distance(d_sd * args.gamma_min, d_sd * args.gamma_max)
            d_rd = d_sd - d_sr

            x_hat, ch_input, loss = tx_relay_model(
                encoder_idx[:, 1:], encoder_attention_mask[:, 1:], d_sr
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        val_losses = []
        tx_relay_model.eval()
        for b in data_handler.val_dataloader:
            encoder_idx = b[0].to(device)
            encoder_attention_mask = b[1].to(device)

            encoder_idx = data_handler.label_encoder.transform(encoder_idx)

            d_sd = get_distance(args.d_min, args.d_max)
            d_sr = get_distance(d_sd * args.gamma_min, d_sd * args.gamma_max)
            d_rd = d_sd - d_sr

            with torch.no_grad():
                x_hat, ch_input, loss = tx_relay_model(
                    encoder_idx[:, 1:], encoder_attention_mask[:, 1:], d_sr
                )

            val_losses.append(loss.item())

        print("\n")
        print("Epoch: " + str(epoch))
        print_loss(train_losses, "Train")
        print_loss(val_losses, "Val")

        mean_loss = np.mean(val_losses)
        checkpoint_path = os.path.join(
            args.checkpoint_path,
            f"baseline-tx-relay/baseline_tx_relay_{args.channel_type}_{epoch}.pt",
        )

        if mean_loss < best_loss:
            create_checkpoint(
                path=checkpoint_path,
                model_state_dict=tx_relay_model.state_dict(),
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
