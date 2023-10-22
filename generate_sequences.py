import torch

from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.model.decoder import Decoder


def generate_text():
    model.eval()
    xb, encoder_output = next(iter(data_handler.val_dataloader))
    B, T = xb.shape

    # TODO: write this smarter
    for j in range(B):
        for i in range(T - 2):
            input_token = data_handler.get_tokens(ids=xb[[j], : i + 1])
            next_token = data_handler.get_tokens(ids=xb[[j], i + 1])

            predicted_id = model.generate(
                encoder_output=encoder_output[[j], : i + 1, :],
                sample=False,
            )
            predicted_token = data_handler.get_tokens(ids=predicted_id)
            print(f"{input_token[0]} -> {predicted_token[0]} ({next_token[0]})")

        print("\n")


if __name__ == "__main__":
    device = torch.device("cpu")

    data_handler = DataHandler(device=device)
    data_handler.load_data()

    model = Decoder(
        vocab_size=data_handler.vocab_size,
        n_heads=4,
        n_embeddings=384,
        block_size=data_handler.max_length,
        device=device,
    )
    model.load_state_dict(torch.load("model.pt"))

    with torch.no_grad():
        generate_text()
