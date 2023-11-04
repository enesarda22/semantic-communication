import argparse
import os

import numpy as np
from tqdm import tqdm
import torch

from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.models.semantic_decoder import SemanticDecoder
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.utils.general import (
    get_device,
    print_loss,
    create_checkpoint,
    set_seed,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model args
    parser.add_argument("--relay-decoder-path", default=None, type=str)
    parser.add_argument("--n-blocks", default=1, type=int)
    parser.add_argument("--n-heads", default=4, type=int)
    parser.add_argument("--n-embeddings", default=384, type=int)

    # data args
    parser.add_argument("--max-length", default=30, type=int)
    parser.add_argument("--data-fp", default="", type=str)

    # train args
    parser.add_argument("--n-epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--checkpoint-path", default="checkpoints", type=str)
    args = parser.parse_args()

    set_seed()
    device = get_device()

    semantic_encoder = SemanticEncoder(max_length=args.max_length)
    data_handler = DataHandler(
        semantic_encoder=semantic_encoder,
        batch_size=args.batch_size,
        data_fp=args.data_fp,
    )

    relay_decoder = SemanticDecoder(
        vocab_size=data_handler.vocab_size,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        n_embeddings=args.n_embeddings,
        block_size=args.max_length,
    ).to(device)
    optimizer = torch.optim.AdamW(relay_decoder.parameters(), lr=args.lr)

    if args.relay_decoder_path is not None:
        checkpoint = torch.load(args.relay_decoder_path)
        relay_decoder.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    best_loss = torch.inf
    for epoch in range(args.n_epochs):
        train_losses = []
        relay_decoder.train()
        for b in tqdm(data_handler.train_dataloader):
            xb = b[0].to(device)
            attention_mask = b[1].to(device)

            encoder_output = semantic_encoder(
                input_ids=xb,
                attention_mask=attention_mask,
            )

            xb = data_handler.encode_token_ids(xb)
            logits, loss = relay_decoder(
                encoder_output=encoder_output[:, :-1, :],
                attention_mask=attention_mask[:, :-1],
                targets=xb[:, 1:],
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        val_losses = []
        relay_decoder.eval()
        for b in data_handler.val_dataloader:
            xb = b[0].to(device)
            attention_mask = b[1].to(device)

            encoder_output = semantic_encoder(
                input_ids=xb,
                attention_mask=attention_mask,
            )
            xb = data_handler.encode_token_ids(xb)

            with torch.no_grad():
                _, loss = relay_decoder(
                    encoder_output=encoder_output[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    targets=xb[:, 1:],
                )
            val_losses.append(loss.item())

        print("\n")
        print_loss(train_losses, "Train")
        print_loss(val_losses, "Val")

        mean_loss = np.mean(val_losses)

        checkpoint_path = os.path.join(
            args.checkpoint_path,
            f"relay-decoder/relay_decoder_{epoch}.pt",
        )

        if mean_loss < best_loss:
            create_checkpoint(
                path=checkpoint_path,
                model_state_dict=relay_decoder.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                mean_val_loss=mean_loss,
            )
            best_loss = mean_loss
        else:
            create_checkpoint(
                path=checkpoint_path,
                model_state_dict=None,
                optimizer_state_dict=None,
                mean_val_loss=mean_loss,
            )
