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
from semantic_communication.utils.channel import AWGN, Rayleigh
from semantic_communication.models.semantic_decoder import SemanticDecoder
from semantic_communication.utils.general import (
    get_device,
    print_loss,
    create_checkpoint,
    set_seed,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model args
    parser.add_argument("--relay-decoder-path", type=str)
    parser.add_argument("--n-blocks", default=1, type=int)
    parser.add_argument("--n-heads", default=4, type=int)
    parser.add_argument("--n-embeddings", default=384, type=int)
    parser.add_argument("--channel-block-input-dim", default=384, type=int)
    parser.add_argument("--channel-block-latent-dim", default=128, type=int)

    # data args
    parser.add_argument("--max-length", default=30, type=int)
    parser.add_argument("--data-fp", default="", type=str)

    # train args
    parser.add_argument("--n-epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--sig-pow", default=1.0, type=float)
    parser.add_argument("--SNR-min", default=3, type=int)
    parser.add_argument("--SNR-max", default=21, type=int)
    parser.add_argument("--SNR-step", default=3, type=int)
    parser.add_argument("--SNR-window", default=5, type=int)
    parser.add_argument("--SNR-diff", default=3, type=int)
    parser.add_argument("--channel-type", default="AWGN", type=str)
    parser.add_argument("--checkpoint-path", default="checkpoints", type=str)
    args = parser.parse_args()

    device = get_device()
    set_seed()
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
    )

    # Initializations
    SNR_dB = np.flip(np.arange(args.SNR_min, args.SNR_max + 1, args.SNR_step))

    if args.channel_type == "AWGN":
        channel_tx_rx = AWGN(SNR_dB[0] - args.SNR_diff, args.sig_pow)
        channel_rel_rx = AWGN(SNR_dB[0], args.sig_pow)
    else:
        channel_tx_rx = Rayleigh(SNR_dB[0] - args.SNR_diff, args.sig_pow)
        channel_rel_rx = Rayleigh(SNR_dB[0], args.sig_pow)

    tx_relay_rx_channel_model = TxRelayRxChannelModel(
        nin=args.channel_block_input_dim,
        n_latent=args.channel_block_latent_dim,
        channel_tx_rx=channel_tx_rx,
        channel_rel_rx=channel_rel_rx,
    ).to(device)

    optimizer = torch.optim.AdamW(
        params=tx_relay_rx_channel_model.parameters(),
        lr=args.lr,
    )
    criterion = torch.nn.MSELoss()

    best_loss = torch.inf
    cur_win, cur_SNR_index = 0, 0

    for epoch in range(args.n_epochs):
        train_losses = []
        tx_relay_rx_channel_model.train()
        if cur_win >= args.SNR_window:
            cur_win = 0
            if not cur_SNR_index >= len(SNR_dB) - 1:
                cur_SNR_index += 1

            if args.channel_type == "AWGN":
                channel_tx_rx = AWGN(
                    SNR_dB[cur_SNR_index] - args.SNR_diff, args.sig_pow
                )
                channel_rel_rx = AWGN(SNR_dB[cur_SNR_index], args.sig_pow)
            else:
                channel_tx_rx = Rayleigh(
                    SNR_dB[cur_SNR_index] - args.SNR_diff, args.sig_pow
                )
                channel_rel_rx = Rayleigh(SNR_dB[cur_SNR_index], args.sig_pow)

            tx_relay_rx_channel_model.channel_tx_rx = channel_tx_rx
            tx_relay_rx_channel_model.channel_rel_rx = channel_rel_rx

        cur_win += 1
        for b in tqdm(data_handler.train_dataloader):
            xb = b[0].to(device)
            attention_mask = b[1].to(device)

            encoder_output = semantic_encoder(
                input_ids=xb,
                attention_mask=attention_mask,
            )
            relay_out = relay(x=encoder_output[:, :-1, :])

            output_hat = tx_relay_rx_channel_model(
                encoder_output[:, 1:, :], relay_out
            )
            loss = criterion(output_hat, encoder_output[:, 1:, :] + relay_out)

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

            with torch.no_grad():
                output_hat = tx_relay_rx_channel_model(
                    encoder_output[:, 1:, :], relay_out
                )

            loss = criterion(output_hat, encoder_output[:, 1:, :] + relay_out)
            val_losses.append(loss.item())

        print("\n")
        print_loss(train_losses, "Train")
        print_loss(val_losses, "Val")

        mean_loss = np.mean(val_losses)

        checkpoint_path = os.path.join(
            args.checkpoint_path,
            f"tx-relay-rx-channel/tx_relay_rx_channel_{epoch}.pt",
        )

        if mean_loss < best_loss:
            create_checkpoint(
                path=checkpoint_path,
                model_state_dict=tx_relay_rx_channel_model.state_dict(),
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
