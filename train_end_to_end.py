import argparse
import os

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import (
    init_process_group,
    destroy_process_group,
    all_gather_object,
)
from tqdm import tqdm

from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.models.semantic_transformer import SemanticTransformer
from semantic_communication.models.transceiver import (
    Transceiver,
    ChannelEncoder,
    ChannelDecoder,
    init_dst_channel_decoder_state_dict,
    init_relay_semantic_encoder_state_dict,
)
from semantic_communication.utils.channel import (
    init_channel,
    get_distance,
)
from semantic_communication.models.semantic_decoder import SemanticDecoder
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.utils.general import (
    get_device,
    print_loss,
    create_checkpoint,
    set_seed,
    add_semantic_decoder_args,
    add_channel_model_args,
    add_data_args,
    add_train_args,
    load_model,
    load_optimizer,
    load_scheduler,
    get_start_epoch,
)


def main(args):
    world_size = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ["LOCAL_RANK"])
    print(world_size, local_rank)

    init_process_group(backend="nccl")
    set_seed(local_rank)
    device = get_device()
    torch.cuda.set_device(local_rank)

    data_handler = DataHandler(
        batch_size=args.batch_size,
        data_fp=args.data_fp,
        rank=local_rank,
        world_size=world_size,
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

    channel_encoder = ChannelEncoder(
        nin=args.channel_block_input_dim,
        nout=args.channel_block_latent_dim,
    ).to(device)

    channel_decoder = ChannelDecoder(
        nin=args.channel_block_latent_dim,
        nout=args.channel_block_input_dim,
    ).to(device)

    channel = init_channel(args.channel_type, args.sig_pow, args.alpha, args.noise_pow)

    semantic_transformer = SemanticTransformer(
        semantic_encoder=semantic_encoder,
        semantic_decoder=semantic_decoder,
        channel_encoder=channel_encoder,
        channel_decoder=channel_decoder,
        channel=channel,
    ).to(device)
    load_model(semantic_transformer, args.semantic_transformer_path)

    relay_semantic_encoder = SemanticEncoder(
        label_encoder=data_handler.label_encoder,
        max_length=args.max_length,
        mode=args.mode if args.mode == "sentence" else "forward",
        rate=1 if args.mode == "sentence" else None,
    ).to(device)
    state_dict = init_relay_semantic_encoder_state_dict(semantic_transformer)
    relay_semantic_encoder.load_state_dict(state_dict)

    relay_channel_encoder = ChannelEncoder(
        nin=args.channel_block_input_dim,
        nout=args.channel_block_latent_dim,
    ).to(device)
    relay_channel_encoder.load_state_dict(
        semantic_transformer.channel_encoder.state_dict()
    )

    dst_channel_decoder = ChannelDecoder(
        nin=args.channel_block_latent_dim * 2,
        nout=args.channel_block_input_dim,
    ).to(device)
    state_dict = init_dst_channel_decoder_state_dict(
        semantic_transformer, mode=args.mode
    )
    dst_channel_decoder.load_state_dict(state_dict, strict=False)

    dst_semantic_decoder = SemanticDecoder(
        vocab_size=data_handler.vocab_size,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        n_embeddings=args.n_embeddings,
        block_size=args.max_length,
        bert=relay_semantic_encoder.bert,
        pad_idx=data_handler.label_encoder.pad_id,
    ).to(device)
    dst_semantic_decoder.load_state_dict(
        semantic_transformer.semantic_decoder.state_dict()
    )

    transceiver = Transceiver(
        src_relay_transformer=semantic_transformer,
        relay_semantic_encoder=relay_semantic_encoder,
        relay_channel_encoder=relay_channel_encoder,
        dst_channel_decoder=dst_channel_decoder,
        dst_semantic_decoder=dst_semantic_decoder,
        channel=channel,
        max_length=args.max_length,
    )
    load_model(transceiver, args.transceiver_path)
    transceiver = DDP(transceiver, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(transceiver.parameters(), lr=args.lr)
    if args.load_optimizer:
        load_optimizer(optimizer, args.transceiver_path)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(data_handler.train_dataloader),
        epochs=args.n_epochs,
    )
    if args.load_scheduler:
        load_scheduler(scheduler, args.transceiver_path)
        start_epoch = get_start_epoch(args.transceiver_path)
    else:
        start_epoch = 1

    best_loss = torch.inf
    for epoch in range(start_epoch, args.n_epochs + 1):
        data_handler.train_dataloader.sampler.set_epoch(epoch)
        train_losses = []
        transceiver.train()

        for b in tqdm(data_handler.train_dataloader):
            encoder_idx = b[0].to(device)
            encoder_attention_mask = b[1].to(device)

            encoder_idx = data_handler.label_encoder.transform(encoder_idx)

            d_sd = get_distance(args.d_min, args.d_max)
            d_sr = get_distance(d_sd * args.gamma_min, d_sd * args.gamma_max)

            _, loss = transceiver(
                input_ids=encoder_idx,
                attention_mask=encoder_attention_mask,
                d_sd=d_sd,
                d_sr=d_sr,
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            train_losses.append(loss.item())

        data_handler.val_dataloader.sampler.set_epoch(epoch)
        val_losses = []
        transceiver.eval()
        for i, b in tqdm(enumerate(data_handler.val_dataloader)):
            encoder_idx = b[0].to(device)
            encoder_attention_mask = b[1].to(device)

            encoder_idx = data_handler.label_encoder.transform(encoder_idx)

            d_sd = get_distance(args.d_min, args.d_max)
            d_sr = get_distance(d_sd * args.gamma_min, d_sd * args.gamma_max)

            with torch.no_grad():
                _, loss = transceiver(
                    input_ids=encoder_idx,
                    attention_mask=encoder_attention_mask,
                    d_sd=d_sd,
                    d_sr=d_sr,
                )
            val_losses.append(loss.item())

            if i >= args.eval_iter:
                break

        all_val_losses = [None for _ in range(world_size)]
        all_gather_object(all_val_losses, val_losses)

        if local_rank == 0:
            print("\n")
            print_loss(train_losses, "Train")
            print_loss(all_val_losses, "Val")

            mean_loss = np.mean(all_val_losses)

            checkpoint_path = os.path.join(
                args.checkpoint_path,
                f"transceiver/transceiver_{epoch}.pt",
            )

            if mean_loss < best_loss:
                create_checkpoint(
                    path=checkpoint_path,
                    model_state_dict=transceiver.module.state_dict(),
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

    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transceiver-path", type=str)
    parser.add_argument("--semantic-transformer-path", default=None, type=str)
    add_semantic_decoder_args(parser)
    add_data_args(parser)
    add_train_args(parser)
    add_channel_model_args(parser)
    args = parser.parse_args()

    main(args=args)
