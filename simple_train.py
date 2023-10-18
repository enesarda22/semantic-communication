import numpy as np
from tqdm import tqdm
import torch

from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.model.decoder import Decoder


def print_loss(losses, group):
    mean_loss = np.mean(losses)
    se = np.std(losses, ddof=1) / np.sqrt(len(losses))
    print(f"{group} Mean Loss: {mean_loss:.3f} ± {se:.3f}")


def generate_text():
    model.eval()
    xb, encoder_output = next(iter(data_handler.val_dataloader))
    generated_sequence = model.generate_from_scratch(
        encoder_output=encoder_output,
        sample=False,
    ).flatten()

    generated_tokens = data_handler.encoder.inverse_transform(
        generated_sequence
    ).reshape(xb.shape[0], -1)
    generated_text = [
        data_handler.tokenizer.decode(tokens) for tokens in generated_tokens
    ]

    actual_tokens = data_handler.encoder.inverse_transform(xb.flatten()).reshape(
        xb.shape[0], -1
    )
    actual_text = [data_handler.tokenizer.decode(tokens) for tokens in actual_tokens]

    for generated, actual in zip(generated_text, actual_text):
        print(f"Generated Text: {generated}")
        print(f"Actual Text: {actual}\n")


if __name__ == "__main__":
    device = torch.device("cpu")

    data_handler = DataHandler(device=device)
    data_handler.load_data(with_encoder_output=True)

    model = Decoder(
        vocab_size=data_handler.vocab_size,
        n_heads=4,
        n_embeddings=384,
        block_size=data_handler.max_length,
        device=device,
    )
    # generate_text()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for _ in range(15):
        train_losses = []
        model.train()
        for b in tqdm(data_handler.train_dataloader):
            xb = b[0].to(device)
            encoder_output = b[1].to(device)

            logits, loss = model(xb[:, :-1], encoder_output, xb[:, 1:])
            optimizer.zero_grad(set_to_none=True)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        val_losses = []
        model.eval()
        for b in data_handler.val_dataloader:
            xb = b[0].to(device)
            encoder_output = b[1].to(device)

            _, loss = model(xb[:, :-1], encoder_output, xb[:, 1:])
            val_losses.append(loss.item())

        print("\n")
        print_loss(train_losses, "Train")
        print_loss(val_losses, "Val")

    generate_text()
