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
    set_seed,
    add_semantic_decoder_args,
    add_train_args,
    add_data_args,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--relay-decoder-path", type=str)
    parser.add_argument("--receiver-decoder-path", default=None, type=str)

    add_semantic_decoder_args(parser)
    add_data_args(parser)
    add_train_args(parser)
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
    checkpoint = torch.load(args.relay_decoder_path, map_location=device)
    relay_decoder.load_state_dict(checkpoint["model_state_dict"])
    relay_decoder.to(device)

    relay = Relay(
        semantic_encoder=semantic_encoder,
        semantic_decoder=relay_decoder,
        encoder=data_handler.encoder,
    ).to(device)
    receiver_decoder = SemanticDecoder(
        vocab_size=data_handler.vocab_size,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        n_embeddings=args.n_embeddings,
        block_size=args.max_length,
    ).to(device)
    optimizer = torch.optim.AdamW(receiver_decoder.parameters(), lr=args.lr)

    if args.receiver_decoder_path is not None:
        checkpoint = torch.load(args.receiver_decoder_path)
        receiver_decoder.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    best_loss = torch.inf
    for epoch in range(args.n_epochs):
        train_losses = []
        receiver_decoder.train()
        for b in tqdm(data_handler.train_dataloader):
            xb = b[0].to(device)
            attention_mask = b[1].to(device)

            encoder_output = semantic_encoder(
                input_ids=xb,
                attention_mask=attention_mask,
            )
            relay_out = relay(x=encoder_output[:, :-1, :])
            superposed_out = relay_out + encoder_output[:, 1:, :]

            xb = data_handler.encode_token_ids(xb)
            logits, loss = receiver_decoder(
                encoder_output=superposed_out,
                attention_mask=attention_mask[:, 1:],
                targets=xb[:, 1:],
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        val_losses = []
        receiver_decoder.eval()
        for b in data_handler.val_dataloader:
            xb = b[0].to(device)
            attention_mask = b[1].to(device)

            encoder_output = semantic_encoder(
                input_ids=xb,
                attention_mask=attention_mask,
            )
            relay_out = relay(x=encoder_output[:, :-1, :])
            superposed_out = relay_out + encoder_output[:, 1:, :]

            xb = data_handler.encode_token_ids(xb)
            with torch.no_grad():
                _, loss = receiver_decoder(
                    encoder_output=superposed_out,
                    attention_mask=attention_mask[:, 1:],
                    targets=xb[:, 1:],
                )
            val_losses.append(loss.item())

        print("\n")
        print_loss(train_losses, "Train")
        print_loss(val_losses, "Val")

        mean_loss = np.mean(val_losses)

        checkpoint_path = os.path.join(
            args.checkpoint_path,
            f"receiver-decoder/receiver_decoder_{epoch}.pt",
        )

        if mean_loss < best_loss:
            create_checkpoint(
                path=checkpoint_path,
                model_state_dict=receiver_decoder.state_dict(),
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
