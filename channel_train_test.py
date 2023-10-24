import numpy as np
from tqdm import tqdm
import torch

from semantic_communication.models.channel_enc_dec import *
from semantic_communication.utils.channel import *
from semantic_communication.data_processing.data_handler import DataHandler


def print_loss(losses, group):
    mean_loss = np.mean(losses)
    se = np.std(losses, ddof=1) / np.sqrt(len(losses))
    print(f"{group} Mean Loss: {mean_loss:.3f} ± {se:.3f}")


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_handler = DataHandler(device=device)
    data_handler.load_data()

    # TX - RELAY
    class tx_relay(nn.Module):
        def __init__(self, SNR, sig_pow):
            super(tx_relay, self).__init__()

            self.tx_encoder = Channel_Encoder(364, 128)

            self.relay_decoder = Channel_Decoder(128, 364)
            self.channel = AWGN(SNR, sig_pow)

        def forward(self, x):
            return self.relay_decoder(self.channel(self.tx_encoder(x)))


    tx_relay_model = tx_relay(1, 1)
    optimizer = torch.optim.AdamW(tx_relay_model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    for _ in range(10):
        train_losses = []
        tx_relay_model.train()
        for b in tqdm(data_handler.train_dataloader):
            xb = b[0].to(device)
            encoder_output = b[1].to(device)

            model_output = tx_relay_model(encoder_output)

            loss = criterion(model_output, encoder_output)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_losses = []
        tx_relay_model.eval()
        for b in data_handler.val_dataloader:
            xb = b[0].to(device)
            encoder_output = b[1].to(device)

            model_output = tx_relay_model(encoder_output)
            loss = criterion(model_output, encoder_output)
            val_losses.append(loss.item())

        print("\n")
        print_loss(train_losses, "Train")
        print_loss(val_losses, "Val")

    # torch.save(model.state_dict(), "tx_relay_model.pt")

    # TX-ENC, RELAY-ENC - RX-DEC
    class tx_relay_rx(nn.Module):
        def __init__(self, SNR_TX_RX, SNR_REL_RX, sig_pow_TX_RX, sig_pow_REL_RX):
            super(tx_relay_rx, self).__init__()

            self.tx_encoder = Channel_Encoder(364, 128)
            self.tx_encoder.load_state_dict(tx_relay_model.tx_encoder.state_dict())  # load decoder of TX enc

            self.relay_encoder = Channel_Encoder(364, 128)

            self.rx_decoder = Channel_Decoder(128, 364)

            self.channel_tx_rx = AWGN(SNR_TX_RX, sig_pow_TX_RX)
            self.channel_rel_rx = AWGN(SNR_REL_RX, sig_pow_REL_RX)

        def forward(self, x1, x2):
            y1 = self.channel_tx_rx(self.tx_encoder(x1))
            y2 = self.channel_rel_rx(self.relay_encoder(x2))

            return self.rx_decoder(y1+y2)


    tx_relay_rx_model = tx_relay_rx(1, 1,1, 1)
    optimizer = torch.optim.AdamW(tx_relay_rx_model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    for _ in range(10):
        train_losses = []
        tx_relay_rx_model.train()
        for b in tqdm(data_handler.train_dataloader):
            xb = b[0].to(device)
            encoder_output = b[1].to(device)

            model_output = tx_relay_rx_model(encoder_output)

            loss = criterion(model_output, encoder_output)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_losses = []
        tx_relay_rx_model.eval()
        for b in data_handler.val_dataloader:
            xb = b[0].to(device)
            encoder_output = b[1].to(device)

            model_output = tx_relay_rx_model(encoder_output)
            loss = criterion(model_output, encoder_output)
            val_losses.append(loss.item())

        print("\n")
        print_loss(train_losses, "Train")
        print_loss(val_losses, "Val")

    # torch.save(model.state_dict(), "relay_rx_model.pt")






