import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.models.semantic_decoder import SemanticDecoder
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.models.transceiver import Relay
from semantic_communication.utils.general import (
    get_device,
    print_loss,
    create_checkpoint,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--relay-decoder-path", type=str)
    parser.add_argument("--checkpoint-path", default="checkpoints", type=str)
    parser.add_argument("--n-samples", default=10000, type=int)
    parser.add_argument("--train-size", default=0.8, type=float)
    parser.add_argument("--max-length", default=30, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--n-epochs", default=10, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--n-heads", default=4, type=int)
    parser.add_argument("--n-embeddings", default=384, type=int)
    args = parser.parse_args()

    device = get_device()

    semantic_encoder = SemanticEncoder(max_length=args.max_length)
    data_handler = DataHandler(
        semantic_encoder=semantic_encoder,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        train_size=args.train_size,
    )
    data_handler.load_data()

    relay_decoder = SemanticDecoder(
        vocab_size=data_handler.vocab_size,
        n_heads=args.n_heads,
        n_embeddings=args.n_embeddings,
        block_size=args.max_length,
    ).to(device)
    checkpoint = torch.load(args.relay_decoder_path)
    relay_decoder.load_state_dict(checkpoint["model_state_dict"])

    relay = Relay(
        semantic_encoder=semantic_encoder,
        semantic_decoder=relay_decoder,
    )
    receiver_decoder = SemanticDecoder(
        vocab_size=data_handler.vocab_size,
        n_heads=args.n_heads,
        n_embeddings=args.n_embeddings,
        block_size=args.max_length,
    ).to(device)
    optimizer = torch.optim.AdamW(receiver_decoder.parameters(), lr=args.lr)

    best_loss = 5
    for epoch in range(args.n_epochs):
        train_losses = []
        receiver_decoder.train()
        for b in tqdm(data_handler.train_dataloader):
            xb = b[0].to(device)
            attention_mask = b[1].to(device)

            encoder_output = semantic_encoder(input_ids=xb, attention_mask=attention_mask)
            relay_out = relay(encoder_output[:, :-1, :], attention_mask[:, :-1, :])
            superposed_out = relay_out + encoder_output[:, 1:, :]

            logits, loss = receiver_decoder(superposed_out, xb[:, 1:])
            optimizer.zero_grad(set_to_none=True)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        val_losses = []
        receiver_decoder.eval()
        for b in data_handler.val_dataloader:
            xb = b[0].to(device)
            attention_mask = b[1].to(device)

            encoder_output = semantic_encoder(input_ids=xb, attention_mask=attention_mask)
            relay_out = relay(encoder_output[:, :-1, :], attention_mask[:, :-1, :])
            superposed_out = relay_out + encoder_output[:, 1:, :]

            with torch.no_grad():
                _, loss = receiver_decoder(superposed_out, xb[:, 1:])
            val_losses.append(loss.item())

        print("\n")
        print_loss(train_losses, "Train")
        print_loss(val_losses, "Val")

        mean_loss = np.mean(val_losses)
        if mean_loss < best_loss:
            create_checkpoint(
                path=os.path.join(
                    args.checkpoint_path,
                    f"receiver-decoder/receiver_decoder_{epoch}.pt",
                ),
                model_state_dict=receiver_decoder.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                mean_val_loss=mean_loss,
            )
            best_loss = mean_loss
