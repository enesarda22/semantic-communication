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


    Relay_channel_dec = Channel_Decoder(128, 364)
    Relay_channel_enc = Channel_Encoder(364, 128)


    # TX - RX
    class tx_rx(nn.Module):
        def __init__(self, SNR, sig_pow):
            super(tx_rx, self).__init__()

            self.encoder = Channel_Encoder(364, 128)
            self.decoder = Channel_Decoder(128,364)
            self.channel = AWGN(SNR, sig_pow)

        def forward(self, x):
            return self.decoder(self.channel(self.encoder(x)))

    tx_rx_model = tx_rx(1, 1)
    optimizer = torch.optim.AdamW(tx_rx_model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    for _ in range(10):
        train_losses = []
        tx_rx_model.train()
        for b in tqdm(data_handler.train_dataloader):
            xb = b[0].to(device)
            encoder_output = b[1].to(device)

            model_output = tx_rx_model(encoder_output)

            loss = criterion(model_output, encoder_output)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_losses = []
        tx_rx_model.eval()
        for b in data_handler.val_dataloader:
            xb = b[0].to(device)
            encoder_output = b[1].to(device)

            model_output = tx_rx_model(encoder_output)
            loss = criterion(model_output, encoder_output)
            val_losses.append(loss.item())

        print("\n")
        print_loss(train_losses, "Train")
        print_loss(val_losses, "Val")

    # torch.save(model.state_dict(), "tx_rx_model.pt")

    # TX - RELAY
    class tx_relay(nn.Module):
        def __init__(self, SNR, sig_pow):
            super(tx_relay, self).__init__()

            self.encoder = Channel_Encoder(364, 128)
            self.encoder.load_state_dict(tx_rx_model.encoder.state_dict())  # load encoder of TX

            self.decoder = Channel_Decoder(128, 364)
            self.channel = AWGN(SNR, sig_pow)

        def forward(self, x):
            return self.decoder(self.channel(self.encoder(x)))


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

    # RELAY - RX
    class relay_rx(nn.Module):
        def __init__(self, SNR, sig_pow):
            super(relay_rx, self).__init__()

            self.encoder = Channel_Encoder(364, 128)

            self.encoder.load_state_dict(tx_rx_model.decoder.state_dict())  # load decoder of RX
            self.channel = AWGN(SNR, sig_pow)

        def forward(self, x):
            return self.decoder(self.channel(self.encoder(x)))


    relay_rx_model = relay_rx(1, 1)
    optimizer = torch.optim.AdamW(relay_rx_model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    for _ in range(10):
        train_losses = []
        relay_rx_model.train()
        for b in tqdm(data_handler.train_dataloader):
            xb = b[0].to(device)
            encoder_output = b[1].to(device)

            model_output = relay_rx_model(encoder_output)

            loss = criterion(model_output, encoder_output)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_losses = []
        relay_rx_model.eval()
        for b in data_handler.val_dataloader:
            xb = b[0].to(device)
            encoder_output = b[1].to(device)

            model_output = relay_rx_model(encoder_output)
            loss = criterion(model_output, encoder_output)
            val_losses.append(loss.item())

        print("\n")
        print_loss(train_losses, "Train")
        print_loss(val_losses, "Val")

    # torch.save(model.state_dict(), "relay_rx_model.pt")






