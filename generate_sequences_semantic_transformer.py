import argparse

from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.models.semantic_decoder import SemanticDecoder
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.models.semantic_transformer import SemanticTransformer
from semantic_communication.utils.general import (
    add_semantic_decoder_args,
    add_data_args,
    set_seed,
    get_device,
    load_model,
)


def generate_text():
    for b in data_handler.test_dataloader:
        encoder_idx = b[0].to(device)
        encoder_attention_mask = b[1].to(device)

        encoder_idx = data_handler.label_encoder.transform(encoder_idx)

        predicted_ids = semantic_transformer.generate(
            input_ids=encoder_idx,
            attention_mask=encoder_attention_mask,
            max_length=args.max_length,
        )

        predicted_tokens = semantic_encoder.get_tokens(
            ids=predicted_ids,
            skip_special_tokens=True,
        )

        input_tokens = semantic_encoder.get_tokens(
            ids=encoder_idx,
            skip_special_tokens=True,
        )

        for input_, predicted in zip(input_tokens, predicted_tokens):
            print(f"{input_}\n{predicted}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-transformer-path", type=str)
    parser.add_argument("--batch-size", default=32, type=int)
    add_semantic_decoder_args(parser)
    add_data_args(parser)
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
    ).to(device)

    semantic_decoder = SemanticDecoder(
        vocab_size=data_handler.vocab_size,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        n_embeddings=args.n_embeddings,
        block_size=args.max_length,
        bert=semantic_encoder.bert,
    ).to(device)

    semantic_transformer = SemanticTransformer(
        semantic_encoder=semantic_encoder,
        semantic_decoder=semantic_decoder,
    ).to(device)
    load_model(semantic_transformer, args.semantic_transformer_path)

    generate_text()
