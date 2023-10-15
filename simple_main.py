from tqdm import tqdm
import torch

from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.model.bigram_language_model import BigramLanguageModel


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
        n_embeddings=32,
        block_size=data_handler.max_length,
        device=device,
    )
    generate_text()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in tqdm(range(1000)):
        for b in data_handler.train_dataloader:
            xb = b[0].to(device)

            logits, loss = model(xb[:, :-1], xb[:, 1:])
            optimizer.zero_grad(set_to_none=True)

            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            print(f"Loss after epoch {epoch}={loss.item()}")

    generate_text()
