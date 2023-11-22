import argparse
from tqdm import tqdm

from semantic_communication.models.baseline_models import Tx_Relay
from semantic_communication.utils.channel import init_channel, get_SNR
from semantic_communication.models.semantic_encoder import SemanticEncoder
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

    semantic_encoder = SemanticEncoder(max_length=args.max_length)
    data_handler = DataHandler(
        semantic_encoder=semantic_encoder,
        batch_size=args.batch_size,
        data_fp=args.data_fp,
    )

    channel = init_channel(args.channel_type, args.sig_pow)
    num_classes = data_handler.vocab_size
    tx_relay_model = Tx_Relay(num_classes, n_emb=args.channel_block_input_dim, n_latent=args.channel_block_latent_dim, channel=channel).to(device)

    optimizer = torch.optim.AdamW(
        params=tx_relay_model.parameters(),
        lr=args.lr,
    )

    for epoch in range(args.n_epochs):
        train_losses = []
        tx_relay_model.train()
        for b in tqdm(data_handler.train_dataloader):
            xb = b[0].to(device)
            attention_mask = b[1].to(device)

            xb = data_handler.encode_token_ids(xb)
            SNR = get_SNR(args.SNR_min, args.SNR_max)

            x_hat, ch_input, loss = tx_relay_model(xb[:, 1:],  attention_mask[:, 1:], SNR)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_losses = []
        tx_relay_model.eval()
        for b in data_handler.val_dataloader:
            xb = b[0].to(device)
            attention_mask = b[1].to(device)

            xb = data_handler.encode_token_ids(xb)
            SNR = get_SNR(args.SNR_min, args.SNR_max)

            with torch.no_grad():
                x_hat, ch_input, loss = tx_relay_model(xb[:, 1:], attention_mask[:, 1:], SNR)

            val_losses.append(loss.item())

        print("\n")
        print("Epoch: " + str(epoch))
        print_loss(train_losses, "Train")
        print_loss(val_losses, "Val")

        mean_loss = np.mean(val_losses)

        create_checkpoint(
            path=os.path.join(
                args.checkpoint_path,
                f"baseline-tx-relay/baseline_tx_relay_{epoch}.pt",
            ),
            model_state_dict=tx_relay_model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            mean_val_loss=mean_loss,
        )