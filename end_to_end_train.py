import torch
from tqdm import tqdm
from torch import nn
import numpy as np

from semantic_communication.utils.channel import AWGN, Rayleigh
from semantic_communication.models.transceiver import Transceiver
from semantic_communication.models.transceiver import (
    TxRelayChannelModel,
    TxRelayRxChannelModel,
)
from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.models.semantic_decoder import SemanticDecoder
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.utils.general import get_device, print_loss
from transformers import AutoModel


if __name__ == "__main__":
    device = get_device()

    semantic_encoder = SemanticEncoder(max_length=10)
    data_handler = DataHandler(semantic_encoder=semantic_encoder)
    data_handler.load_data()

    # Initializations
    sig_pow = 1.0
    SNR_dB = np.flip(np.arange(3, 24, 3))
    SNR_window = 5
    tx_rx_SNR_diff = 3

    # TX-RX Channel
    tx_rx_channel = AWGN(SNR_dB[0] - tx_rx_SNR_diff, sig_pow)

    # TX-Relay Channel
    tx_relay_channel = AWGN(SNR_dB[0], sig_pow)

    # Relay-RX Channel
    relay_rx_channel = AWGN(SNR_dB[0], sig_pow)

    bert = AutoModel.from_pretrained(data_handler.model_name).to(device)

    relay_semantic_decoder = SemanticDecoder(
        vocab_size=data_handler.vocab_size,
        n_heads=4,
        n_embeddings=384,
        block_size=data_handler.max_length,
        device=device,
    ).to(device)

    rx_semantic_decoder = SemanticDecoder(
        vocab_size=data_handler.vocab_size,
        n_heads=4,
        n_embeddings=384,
        block_size=data_handler.max_length,
        device=device,
    ).to(device)

    # Load from pre_trained
    relay_semantic_decoder.load_state_dict(torch.load('relay_decoder.pt'))
    rx_semantic_decoder.load_state_dict(torch.load("receiver_decoder.pt"))

    tx_relay_channel_enc_dec = TxRelayChannelModel(384, 128, tx_relay_channel)
    tx_relay_rx_channel_enc_dec = TxRelayRxChannelModel(384, 128, tx_rx_channel, relay_rx_channel)

    # Load from pre_trained
    tx_relay_channel_enc_dec.load_state_dict(torch.load('tx_relay_channel_enc_dec.pt'))
    tx_relay_rx_channel_enc_dec.load_state_dict(torch.load('tx_relay_rx_channel_enc_dec.pt'))

    transceiver = Transceiver(tx_relay_channel, relay_rx_channel, tx_rx_channel, bert, relay_semantic_decoder,
                              rx_semantic_decoder, tx_relay_channel_enc_dec, tx_relay_rx_channel_enc_dec)

    optimizer = torch.optim.AdamW(transceiver.parameters(), lr=1e-4)

    cur_win, cur_SNR_index = 0, 0

    for _ in range(10):
        train_losses = []
        transceiver.train()
        for b in tqdm(data_handler.train_dataloader):
            xb = b[0].to(device)

            if cur_win > SNR_window:
                cur_win = 0,
                if not cur_SNR_index >= len(SNR_dB):
                    cur_SNR_index += 1

                # TX-RX Channel
                tx_rx_channel = AWGN(SNR_dB[cur_SNR_index] - tx_rx_SNR_diff, sig_pow)

                # TX-Relay Channel
                tx_relay_channel = AWGN(SNR_dB[cur_SNR_index], sig_pow)

                # Relay-RX Channel
                relay_rx_channel = AWGN(SNR_dB[cur_SNR_index], sig_pow)

                tx_relay_channel_enc_dec.channel = tx_relay_channel
                tx_relay_rx_channel_enc_dec.channel_tx_rx = tx_rx_channel
                tx_relay_rx_channel_enc_dec.channel_rel_rx = relay_rx_channel

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
                _, loss = transceiver(xb, xb[:, 1:])
            val_losses.append(loss.item())

        print("\n")
        print_loss(train_losses, "Train")
        print_loss(val_losses, "Val")

    torch.save(transceiver.state_dict(), "receiver_decoder.pt")
