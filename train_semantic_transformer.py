import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.models.semantic_decoder import SemanticDecoder
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.models.semantic_transformer import SemanticTransformer
from semantic_communication.utils.general import (
    add_semantic_decoder_args,
    add_data_args,
    add_train_args,
    set_seed,
    get_device,
    load_model,
    load_optimizer,
    load_scheduler,
    get_start_epoch,
    create_checkpoint,
    print_loss,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--semantic-decoder-path", default=None, type=str)
    parser.add_argument("--semantic-transformer-path", default=None, type=str)
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

    semantic_decoder = SemanticDecoder(
        vocab_size=data_handler.vocab_size,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        n_embeddings=args.n_embeddings,
        block_size=args.max_length,
        semantic_encoder=semantic_encoder,
        label_encoder=data_handler.label_encoder,
    ).to(device)
    load_model(semantic_decoder, args.semantic_decoder_path)

    semantic_transformer = SemanticTransformer(
        semantic_encoder=semantic_encoder.bert,
        semantic_decoder=semantic_decoder,
        mode=args.mode,
    )
    load_model(semantic_transformer, args.semantic_transformer_path)

    optimizer = torch.optim.AdamW(semantic_transformer.parameters(), lr=args.lr)
    load_optimizer(optimizer, args.semantic_transformer_path)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(data_handler.train_dataloader),
        epochs=args.n_epochs,
    )
    load_scheduler(scheduler, args.semantic_transformer_path)

    start_epoch = get_start_epoch(args.semantic_transformer_path)
    best_loss = torch.inf
    for epoch in range(start_epoch, args.n_epochs + 1):
        train_losses = []
        semantic_transformer.train()
        for b in tqdm(data_handler.train_dataloader):
            encoder_idx = b[0].to(device)
            encoder_attention_mask = b[1].to(device)

            decoder_idx = data_handler.label_encoder.transform(encoder_idx)

            _, loss = semantic_transformer(
                decoder_idx=decoder_idx,
                encoder_idx=encoder_idx,
                encoder_attention_mask=encoder_attention_mask,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())

        val_losses = []
        semantic_transformer.eval()
        for b in data_handler.val_dataloader:
            encoder_idx = b[0].to(device)
            encoder_attention_mask = b[1].to(device)

            decoder_idx = data_handler.label_encoder.transform(encoder_idx)

            with torch.no_grad():
                _, loss = semantic_transformer(
                    decoder_idx=decoder_idx,
                    encoder_idx=encoder_idx,
                    encoder_attention_mask=encoder_attention_mask,
                )
            val_losses.append(loss.item())

        print("\n")
        print_loss(train_losses, "Train")
        print_loss(val_losses, "Val")

        mean_loss = np.mean(val_losses)

        checkpoint_path = os.path.join(
            args.checkpoint_path,
            f"semantic-transformer/semantic_transformer_{epoch}.pt",
        )

        if mean_loss < best_loss:
            create_checkpoint(
                path=checkpoint_path,
                model_state_dict=semantic_transformer.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                mean_val_loss=mean_loss,
                epoch=epoch,
            )
            best_loss = mean_loss
        else:
            create_checkpoint(
                path=checkpoint_path,
                model_state_dict=None,
                optimizer_state_dict=None,
                scheduler_state_dict=None,
                mean_val_loss=mean_loss,
                epoch=epoch,
            )
