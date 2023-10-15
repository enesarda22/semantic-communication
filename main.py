from tqdm import tqdm

import torch
import torch.nn.functional as F

from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.model.net import Net


if __name__ == "__main__":
    device = torch.device("cpu")

    data_handler = DataHandler(device=device)
    data_handler.load_data()

    model = Net(
        embedding_dim=384,
        n_heads=6,
        feedforward_dim=1024,
        dropout=0.1,
        target_vocab_size=data_handler.vocab_size,
        num_layers=8,
        max_length=data_handler.max_length,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_values = []

    for _ in range(5):
        total_loss = 0
        model.train()

        for batch in tqdm(data_handler.train_dataloader):
            xb, encoder_outputs = tuple(t.to(device) for t in batch)

            model.zero_grad()
            logits = model(xb[:, :-1], encoder_outputs)

            B, T, C = logits.shape
            logits = logits.reshape(B * T, C)
            yb = xb[:, 1:].reshape(B * T)

            loss = F.cross_entropy(logits, yb)
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        avg_train_loss = total_loss / len(data_handler.train_dataloader)
        loss_values.append(avg_train_loss)

        print(f"Average train loss: {avg_train_loss:.2f}\n")

        model.eval()
        for batch in data_handler.val_dataloader:
            xb, encoder_outputs = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                logits = model(xb[:, :-1], encoder_outputs)

            input_ids = torch.argmax(logits, dim=-1)

            decoded = [data_handler.tokenizer.decode(x) for x in input_ids]
            actual = [data_handler.tokenizer.decode(x) for x in xb[:, 1:]]

            print(f"Decoded = {decoded[0]}")
            print(f"Actual = {actual[0]}")
