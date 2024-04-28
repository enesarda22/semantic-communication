import argparse
import os

import numpy as np
from tqdm import tqdm
import torch

from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.models.semantic_decoder import SemanticDecoder
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.models.semantic_transformer import SemanticTransformer
from semantic_communication.models.transceiver import (
    ChannelEncoder,
    ChannelDecoder,
    SrcRelayBlock,
)

from semantic_communication.utils.channel import init_channel, get_distance
from semantic_communication.utils.general import (
    get_device,
    print_loss,
    create_checkpoint,
    set_seed,
    add_semantic_decoder_args,
    add_data_args,
    add_train_args,
    add_channel_model_args,
    load_model,
    load_optimizer,
    load_scheduler,
    get_start_epoch,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--semantic-transformer-path", default=None, type=str)
    parser.add_argument("--src-relay-block-path", default=None, type=str)
    add_semantic_decoder_args(parser)
    add_data_args(parser)
    add_train_args(parser)
    add_channel_model_args(parser)
    args = parser.parse_args()

    set_seed()
    device = get_device()

    data_handler = DataHandler(
        batch_size=args.batch_size,
        data_fp=args.data_fp,
    )

    semantic_encoder = SemanticEncoder(
        label_encoder=data_handler.label_encoder,
        max_length=args.max_length,
        mode=args.mode,
        rate=args.rate,
    ).to(device)

    semantic_decoder = SemanticDecoder(
        vocab_size=data_handler.vocab_size,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        n_embeddings=args.n_embeddings,
        block_size=args.max_length,
        bert=semantic_encoder.bert,
        pad_idx=data_handler.label_encoder.pad_id,
    ).to(device)

    semantic_transformer = SemanticTransformer(
        semantic_encoder=semantic_encoder,
        semantic_decoder=semantic_decoder,
    ).to(device)
    load_model(semantic_transformer, args.semantic_transformer_path)

    src_channel_encoder = ChannelEncoder(
        nin=args.channel_block_input_dim,
        nout=args.channel_block_latent_dim,
    ).to(device)

    relay_channel_decoder = ChannelDecoder(
        nin=args.channel_block_latent_dim,
        nout=args.channel_block_input_dim,
    ).to(device)

    channel = init_channel(args.channel_type, args.sig_pow, args.alpha, args.noise_pow)

    src_relay_block = SrcRelayBlock(
        semantic_transformer=semantic_transformer,
        src_channel_encoder=src_channel_encoder,
        relay_channel_decoder=relay_channel_decoder,
        channel=channel,
    ).to(device)
    load_model(src_relay_block, args.src_relay_block_path)

    optimizer = torch.optim.AdamW(src_relay_block.parameters(), lr=args.lr)
    if args.load_optimizer:
        load_optimizer(optimizer, args.src_relay_block_path)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(data_handler.train_dataloader),
        epochs=args.n_epochs,
    )
    if args.load_scheduler:
        load_scheduler(scheduler, args.src_relay_block_path)

    start_epoch = get_start_epoch(args.src_relay_block_path)
    best_loss = torch.inf
    for epoch in range(start_epoch, args.n_epochs + 1):
        train_losses = []
        src_relay_block.train()
        for b in tqdm(data_handler.train_dataloader):
            encoder_idx = b[0].to(device)
            encoder_attention_mask = b[1].to(device)

            encoder_idx = data_handler.label_encoder.transform(encoder_idx)

            d_sd = get_distance(args.d_min, args.d_max)
            d_sr = get_distance(d_sd * args.gamma_min, d_sd * args.gamma_max)

            _, loss = src_relay_block(
                input_ids=encoder_idx,
                attention_mask=encoder_attention_mask,
                d_sr=d_sr,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())

        val_losses = []
        src_relay_block.eval()
        for b in data_handler.val_dataloader:
            encoder_idx = b[0].to(device)
            encoder_attention_mask = b[1].to(device)

            encoder_idx = data_handler.label_encoder.transform(encoder_idx)

            d_sd = get_distance(args.d_min, args.d_max)
            d_sr = get_distance(d_sd * args.gamma_min, d_sd * args.gamma_max)

            with torch.no_grad():
                _, loss = src_relay_block(
                    input_ids=encoder_idx,
                    attention_mask=encoder_attention_mask,
                    d_sr=d_sr,
                )
            val_losses.append(loss.item())

        print("\n")
        print_loss(train_losses, "Train")
        print_loss(val_losses, "Val")

        mean_loss = np.mean(val_losses)

        checkpoint_path = os.path.join(
            args.checkpoint_path,
            f"src-relay-block/src_relay_block_{epoch}.pt",
        )

        if mean_loss < best_loss:
            create_checkpoint(
                path=checkpoint_path,
                model_state_dict=src_relay_block.state_dict(),
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
