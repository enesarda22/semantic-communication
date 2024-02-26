import argparse
import os

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.models.paraphrase_detector import ParaphraseDetector
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.utils.general import (
    add_data_args,
    add_train_args,
    add_paraphrase_detector_args,
    get_device,
    load_model,
    load_optimizer,
    load_scheduler,
    get_start_epoch,
    print_loss,
    create_checkpoint,
    set_seed,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--paraphrase-detector-path", default=None, type=str)
    add_paraphrase_detector_args(parser)
    add_data_args(parser)
    add_train_args(parser)
    args = parser.parse_args()

    set_seed()
    device = get_device()

    data_handler = DataHandler(
        batch_size=args.batch_size,
        data_fp=args.data_fp,
        fn_prefix="paraphrase_",
    )

    semantic_encoder = SemanticEncoder(
        label_encoder=data_handler.label_encoder,
        max_length=args.max_length * 2,
        mode="cls",
    ).to(device)

    paraphrase_detector = ParaphraseDetector(
        semantic_encoder=semantic_encoder,
        n_in=args.n_in,
        n_latent=args.n_latent,
    ).to(device)
    load_model(paraphrase_detector, args.paraphrase_detector_path)

    optimizer = torch.optim.AdamW(paraphrase_detector.parameters(), lr=args.lr)
    if args.load_optimizer:
        load_optimizer(optimizer, args.paraphrase_detector_path)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(data_handler.train_dataloader),
        epochs=args.n_epochs,
    )
    if args.load_scheduler:
        load_scheduler(scheduler, args.paraphrase_detector_path)

    criterion = nn.BCEWithLogitsLoss()

    start_epoch = get_start_epoch(args.paraphrase_detector_path)
    best_loss = torch.inf
    for epoch in range(start_epoch, args.n_epochs + 1):
        train_losses = []
        paraphrase_detector.train()
        for b in tqdm(data_handler.train_dataloader):
            encoder_idx = b[0].to(device)
            encoder_attention_mask = b[1].to(device)
            labels = b[2].to(device)

            encoder_idx = data_handler.label_encoder.transform(encoder_idx)
            logits = paraphrase_detector(
                input_ids=encoder_idx,
                attention_mask=encoder_attention_mask,
            )

            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())

        val_losses = []
        paraphrase_detector.eval()
        for b in data_handler.val_dataloader:
            encoder_idx = b[0].to(device)
            encoder_attention_mask = b[1].to(device)
            labels = b[2].to(device)

            encoder_idx = data_handler.label_encoder.transform(encoder_idx)
            with torch.no_grad():
                logits = paraphrase_detector(
                    input_ids=encoder_idx,
                    attention_mask=encoder_attention_mask,
                )
            loss = criterion(logits, labels)
            val_losses.append(loss.item())

        print("\n")
        print_loss(train_losses, "Train")
        print_loss(val_losses, "Val")

        mean_loss = np.mean(val_losses)

        checkpoint_path = os.path.join(
            args.checkpoint_path,
            f"paraphrase-detector/paraphrase_detector_{epoch}.pt",
        )

        if mean_loss < best_loss:
            create_checkpoint(
                path=checkpoint_path,
                model_state_dict=paraphrase_detector.state_dict(),
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
