import numpy as np
from tqdm import tqdm
import torch

from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.model.bigram_language_model import BigramLanguageModel


def print_loss(losses, group):
    mean_loss = np.mean(losses)
    se = np.std(losses, ddof=1) / np.sqrt(len(losses))
    print(f"{group} Mean Loss: {mean_loss:.3f} ± {se:.3f}")


def generate_text():
    model.eval()
    generated_sequence = model.generate_from_scratch().flatten()

    generated_tokens = data_handler.encoder.inverse_transform(generated_sequence)
    generated_text = data_handler.tokenizer.decode(generated_tokens)
    print(generated_text)


if __name__ == "__main__":
    device = torch.device("cpu")

    data_handler = DataHandler(device=device)
    data_handler.load_data()

    model = BigramLanguageModel(
        vocab_size=data_handler.vocab_size,
        n_heads=4,
        n_embeddings=32,
        block_size=data_handler.max_length,
        device=device,
    )
    generate_text()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for _ in range(20):
        train_losses = []
        model.train()
        for b in tqdm(data_handler.train_dataloader):
            xb = b[0].to(device)

            logits, loss = model(xb[:, :-1], xb[:, 1:])
            optimizer.zero_grad(set_to_none=True)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        val_losses = []
        model.eval()
        for b in data_handler.val_dataloader:
            xb = b[0].to(device)

            _, loss = model(xb[:, :-1], xb[:, 1:])
            val_losses.append(loss.item())

        print("\n")
        print_loss(train_losses, "Train")
        print_loss(val_losses, "Val")

    generate_text()
