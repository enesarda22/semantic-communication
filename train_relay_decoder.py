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
    add_semantic_decoder_args,
    add_data_args,
    add_train_args,
    shift_inputs,
    load_model,
    load_optimizer,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--relay-decoder-path", default=None, type=str)
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
        semantic_encoder=semantic_encoder,
        label_encoder=data_handler.label_encoder,
    ).to(device)
    load_model(relay_decoder, args.relay_decoder_path)

    optimizer = torch.optim.AdamW(relay_decoder.parameters(), lr=args.lr)
    load_optimizer(optimizer, args.relay_decoder_path)

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
            xb = data_handler.label_encoder.transform(xb)
            idx, encoder_output, attention_mask, targets = shift_inputs(
                xb=xb,
                encoder_output=encoder_output,
                attention_mask=attention_mask,
                mode=args.mode,
            )
            _, loss = relay_decoder(
                idx=idx,
                encoder_output=encoder_output,
                attention_mask=attention_mask,
                targets=targets,
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
            xb = data_handler.label_encoder.transform(xb)
            idx, encoder_output, attention_mask, targets = shift_inputs(
                xb=xb,
                encoder_output=encoder_output,
                attention_mask=attention_mask,
                mode=args.mode,
            )

            with torch.no_grad():
                _, loss = relay_decoder(
                    idx=idx,
                    encoder_output=encoder_output,
                    attention_mask=attention_mask,
                    targets=targets,
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
