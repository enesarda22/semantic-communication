import torch
from tqdm import tqdm

from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.models.semantic_decoder import SemanticDecoder
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.models.transceiver import Relay
from semantic_communication.utils.general import get_device, print_loss

if __name__ == "__main__":
    device = get_device()

    semantic_encoder = SemanticEncoder(max_length=10)
    data_handler = DataHandler(semantic_encoder=semantic_encoder)
    data_handler.load_data()

    relay_decoder = SemanticDecoder(
        vocab_size=data_handler.vocab_size,
        n_heads=4,
        n_embeddings=384,
        block_size=semantic_encoder.max_length,
    ).to(device)
    relay_decoder.load_state_dict(torch.load("relay_decoder.pt"))
    relay = Relay(
        semantic_encoder=semantic_encoder,
        semantic_decoder=relay_decoder,
    )

    receiver_decoder = SemanticDecoder(
        vocab_size=data_handler.vocab_size,
        n_heads=4,
        n_embeddings=384,
        block_size=semantic_encoder.max_length,
    ).to(device)
    optimizer = torch.optim.AdamW(receiver_decoder.parameters(), lr=1e-4)

    for _ in range(10):
        train_losses = []
        receiver_decoder.train()
        for b in tqdm(data_handler.train_dataloader):
            xb = b[0].to(device)
            encoder_output = b[1].to(device)

            relay_out = relay(encoder_output[:, :-2, :])
            superposed_out = relay_out + encoder_output[:, 1:-1, :]

            logits, loss = receiver_decoder(superposed_out, xb[:, 1:])
            optimizer.zero_grad(set_to_none=True)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        val_losses = []
        receiver_decoder.eval()
        for b in data_handler.val_dataloader:
            xb = b[0].to(device)
            encoder_output = b[1].to(device)

            relay_out = relay(encoder_output[:, :-2, :])
            superposed_out = relay_out + encoder_output[:, 1:-1, :]

            with torch.no_grad():
                _, loss = receiver_decoder(superposed_out, xb[:, 1:])
            val_losses.append(loss.item())

        print("\n")
        print_loss(train_losses, "Train")
        print_loss(val_losses, "Val")

    torch.save(receiver_decoder.state_dict(), "receiver_decoder.pt")
