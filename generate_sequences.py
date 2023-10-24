import numpy as np
import torch

from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.models.semantic_decoder import SemanticDecoder


def generate_text():
    model.eval()
    xb, encoder_output = next(iter(data_handler.val_dataloader))
    B, T = xb.shape

    xb = xb.unsqueeze(1).repeat(1, T, 1)
    mask = torch.tril(torch.ones(B, T, T))
    masked_ids = xb.masked_fill(mask == 0, 0)

    input_tokens = data_handler.get_tokens(ids=masked_ids.reshape(-1, T))
    input_tokens = np.array_split(input_tokens, B)

    predicted_ids = model.generate(
        encoder_output=encoder_output[:, :-2, :],
        sample=False,
    )
    predicted_ids = predicted_ids.unsqueeze(1).repeat(1, T - 1, 1)
    mask = torch.eye(T - 1).unsqueeze(0).repeat(B, 1, 1)
    masked_ids = predicted_ids.masked_fill(mask == 0, 0)

    predicted_tokens = data_handler.get_tokens(ids=masked_ids.reshape(-1, T - 1))
    predicted_tokens = np.array_split(predicted_tokens, B)

    for input_b, predicted_b in zip(input_tokens, predicted_tokens):
        for input_, predicted in zip(input_b, predicted_b):
            print(
                f"{input_.replace('[PAD]', '').strip()} -> "
                f"{predicted.replace('[PAD]', '').strip()}"
            )
        print(f"{input_b[-1].replace('[PAD]', '').strip()}")
        print("\n")


if __name__ == "__main__":
    device = torch.device("cpu")

    data_handler = DataHandler(device=device)
    data_handler.load_data()

    model = SemanticDecoder(
        vocab_size=data_handler.vocab_size,
        n_heads=4,
        n_embeddings=384,
        block_size=data_handler.max_length,
        device=device,
    )
    model.load_state_dict(torch.load("relay_decoder.pt"))

    with torch.no_grad():
        generate_text()
