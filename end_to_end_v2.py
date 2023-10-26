import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.models.transceiver import (TxRelayChannelModel, TxRelayRxChannelModel, Relay, Transceiver)
from semantic_communication.utils.channel import (AWGN, Rayleigh)
from semantic_communication.models.semantic_decoder import SemanticDecoder
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.utils.general import (
    get_device,
    print_loss,
    create_checkpoint,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--relay-decoder-path", type=str)
    parser.add_argument("--receiver-decoder-path", type=str)
    parser.add_argument("--tx-relay-channel-model-path", type=str)
    parser.add_argument("--tx-relay-rx-channel-model-path", type=str)
    parser.add_argument("--checkpoint-path", default="checkpoints", type=str)
    parser.add_argument("--n-samples", default=40000, type=int)
    parser.add_argument("--train-size", default=0.8, type=float)
    parser.add_argument("--max-length", default=10, type=int)
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
    )
    data_handler.load_data()

    # Initializations
    SNR_dB = np.flip(np.arange(args.SNR_min, args.SNR_max, args.SNR_step))

    if args.channel_type == "AWGN":
        tx_rx_channel = AWGN(SNR_dB[0] - args.SNR_diff, args.sig_pow)
        tx_relay_channel = AWGN(SNR_dB[0], args.sig_pow)
        relay_rx_channel = AWGN(SNR_dB[0], args.sig_pow)

    else:
        tx_rx_channel = Rayleigh(SNR_dB[0] - args.SNR_diff, args.sig_pow)
        tx_relay_channel = Rayleigh(SNR_dB[0], args.sig_pow)
        relay_rx_channel = Rayleigh(SNR_dB[0], args.sig_pow)

    relay_decoder = SemanticDecoder(
        vocab_size=data_handler.vocab_size,
        n_heads=args.n_heads,
        n_embeddings=args.n_embeddings,
        block_size=args.max_length,
    ).to(device)
    relay_checkpoint = torch.load(args.relay_decoder_path)
    relay_decoder.load_state_dict(relay_checkpoint["model_state_dict"])

    receiver_decoder = SemanticDecoder(
        vocab_size=data_handler.vocab_size,
        n_heads=args.n_heads,
        n_embeddings=args.n_embeddings,
        block_size=args.max_length,
    ).to(device)

    rx_checkpoint = torch.load(args.receiver_decoder_path)
    receiver_decoder.load_state_dict(rx_checkpoint["model_state_dict"])

    tx_relay_channel_model = TxRelayChannelModel(384, 128, tx_relay_channel)
    tx_relay_rx_channel_model = TxRelayRxChannelModel(384, 128, tx_rx_channel, relay_rx_channel)

    tx_relay_channel_model_checkpoint = torch.load(args.tx_relay_channel_model_path)
    tx_relay_channel_model.load_state_dict(tx_relay_channel_model_checkpoint["model_state_dict"])

    tx_relay_rx_channel_model_checkpoint = torch.load(args.tx_relay_rx_channel_model_path)
    tx_relay_rx_channel_model.load_state_dict(tx_relay_rx_channel_model_checkpoint["model_state_dict"])

    transceiver = Transceiver(semantic_encoder, relay_decoder, receiver_decoder, tx_relay_channel_model, tx_relay_rx_channel_model)

    optimizer = torch.optim.AdamW(transceiver.parameters(), lr=args.lr)

    best_loss = 5
    cur_win, cur_SNR_index = 0, 0

    for epoch in range(args.n_epochs):
        train_losses = []
        transceiver.train()
        for b in tqdm(data_handler.train_dataloader):
            xb = b[0].to(device)

            if cur_win > args.SNR_window:
                cur_win = 0,
                if not cur_SNR_index >= len(SNR_dB):
                    cur_SNR_index += 1

                if args.channel_type == "AWGN":
                    tx_rx_channel = AWGN(SNR_dB[cur_SNR_index] - args.SNR_diff, args.sig_pow)
                    tx_relay_channel = AWGN(SNR_dB[cur_SNR_index], args.sig_pow)
                    relay_rx_channel = AWGN(SNR_dB[cur_SNR_index], args.sig_pow)

                else:
                    tx_rx_channel = Rayleigh(SNR_dB[cur_SNR_index] - args.SNR_diff, args.sig_pow)
                    tx_relay_channel = Rayleigh(SNR_dB[cur_SNR_index], args.sig_pow)
                    relay_rx_channel = Rayleigh(SNR_dB[cur_SNR_index], args.sig_pow)

                transceiver.tx_relay_channel_enc_dec.channel =tx_relay_channel
                transceiver.tx_relay_rx_channel_enc_dec.channel_tx_rx = tx_rx_channel
                transceiver.tx_relay_rx_channel_enc_dec.channel_rel_rx = relay_rx_channel

            cur_win += 1

            logits, loss = transceiver(xb, xb[:, 1:])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_losses = []
        transceiver.eval()
        for b in data_handler.val_dataloader:
            xb = b[0].to(device)

            with torch.no_grad():
                logits, loss = transceiver(xb, xb[:, 1:])

            val_losses.append(loss.item())

        print("\n")
        print_loss(train_losses, "Train")
        print_loss(val_losses, "Val")

        mean_loss = np.mean(val_losses)
        if mean_loss < best_loss:
            create_checkpoint(
                path=os.path.join(
                    args.checkpoint_path,
                    f"end-to-end_transceiver/end_to_end_transceiver_{epoch}.pt",
                ),
                model_state_dict=tx_relay_channel_model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                mean_val_loss=mean_loss,
            )
            best_loss = mean_loss