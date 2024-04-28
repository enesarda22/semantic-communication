import argparse
import numpy as np

from nltk.translate.bleu_score import sentence_bleu
from torch import nn

from semantic_communication.models.semantic_transformer import SemanticTransformer
from semantic_communication.models.transceiver import (
    Transceiver,
    ChannelEncoder,
    ChannelDecoder,
    SrcRelayBlock,
)
from semantic_communication.utils.general import (
    get_device,
    set_seed,
    add_semantic_decoder_args,
    add_channel_model_args,
    add_data_args,
    load_model,
)
from semantic_communication.models.semantic_encoder import SemanticEncoder
from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.models.semantic_decoder import SemanticDecoder
from semantic_communication.utils.channel import init_channel


def bleu_1gram(target_sentences, received_sentences):
    return sentence_bleu([target_sentences], received_sentences, weights=(1, 0, 0, 0))


def bleu_2gram(target_sentences, received_sentences):
    return sentence_bleu([target_sentences], received_sentences, weights=(0, 1, 0, 0))


def bleu_3gram(target_sentences, received_sentences):
    return sentence_bleu([target_sentences], received_sentences, weights=(0, 0, 1, 0))


def bleu_4gram(target_sentences, received_sentences):
    return sentence_bleu([target_sentences], received_sentences, weights=(0, 0, 0, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model args
    parser.add_argument("--transceiver-path", type=str)
    add_semantic_decoder_args(parser)
    add_channel_model_args(parser)
    add_data_args(parser)

    # test args
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--gamma-list", nargs="+", type=float)
    parser.add_argument("--d-list", nargs="+", type=float)
    parser.add_argument("--n-test", default=10000, type=int)

    args = parser.parse_args()
    device = get_device()
    set_seed()

    data_handler = DataHandler(
        batch_size=args.batch_size,
        data_fp=args.data_fp,
    )

    # initialize models
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

    relay_semantic_encoder = SemanticEncoder(
        label_encoder=data_handler.label_encoder,
        max_length=args.max_length,
        mode=args.mode if args.mode == "sentence" else "forward",
        rate=args.rate,
    ).to(device)

    relay_channel_encoder = ChannelEncoder(
        nin=args.channel_block_input_dim,
        nout=args.channel_block_latent_dim,
    ).to(device)

    dst_channel_decoder = ChannelDecoder(
        nin=args.channel_block_latent_dim * 2,
        nout=args.channel_block_input_dim,
    ).to(device)

    dst_semantic_decoder = SemanticDecoder(
        vocab_size=data_handler.vocab_size,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        n_embeddings=args.n_embeddings,
        block_size=args.max_length,
        bert=semantic_encoder.bert,
        pad_idx=data_handler.label_encoder.pad_id,
    ).to(device)

    transceiver = Transceiver(
        src_relay_block=src_relay_block,
        relay_semantic_encoder=relay_semantic_encoder,
        relay_channel_encoder=relay_channel_encoder,
        dst_channel_decoder=dst_channel_decoder,
        dst_semantic_decoder=dst_semantic_decoder,
        channel=channel,
        max_length=args.max_length,
    )
    transceiver = nn.DataParallel(transceiver)  # TODO: remove module. prefix?
    transceiver = transceiver.to(device)
    load_model(transceiver, args.transceiver_path)

    n_d = len(args.d_list)
    n_gamma = len(args.gamma_list)

    mean_bleu_1 = np.zeros((n_d, n_gamma))
    mean_bleu_3 = np.zeros((n_d, n_gamma))

    std_bleu_1 = np.zeros((n_d, n_gamma))
    std_bleu_3 = np.zeros((n_d, n_gamma))

    # for each d_sd
    for distance_index, d_sd in enumerate(args.d_list):
        # for each gamma in gamma list
        for gamma_index, gamma in enumerate(args.gamma_list):
            print(f"Simulating for distance: {d_sd}  - Gamma: {gamma}")

            cosine_scores = []
            bleu1_scores = []
            bleu3_scores = []

            d_sr = d_sd * gamma

            for b in data_handler.test_dataloader:
                encoder_idx = b[0].to(device)
                encoder_attention_mask = b[1].to(device)

                encoder_idx = data_handler.label_encoder.transform(encoder_idx)

                predicted_ids, probs = transceiver.module.generate(
                    input_ids=encoder_idx,
                    attention_mask=encoder_attention_mask,
                    d_sd=d_sd,
                    d_sr=d_sr,
                )

                predicted_tokens = semantic_encoder.get_tokens(
                    ids=predicted_ids,
                    skip_special_tokens=True,
                )

                input_tokens = semantic_encoder.get_tokens(
                    ids=encoder_idx,
                    skip_special_tokens=True,
                )

                for s1, s2 in zip(input_tokens, predicted_tokens):
                    print(f"True Sentence: {s1}\nPredicted Sentence: {s2}\n")

                    bleu1_score = bleu_1gram(s1, s2)
                    bleu3_score = bleu_3gram(s1, s2)

                    bleu1_scores.append(bleu1_score)
                    bleu3_scores.append(bleu3_score)

                if len(cosine_scores) > args.n_test:
                    break

            n_test_samples = len(cosine_scores)

    np.save("spf_mean_bleu_1.npy", mean_bleu_1)
    np.save("spf_mean_bleu_3.npy", mean_bleu_3)

    np.save("spf_std_bleu_1.npy", std_bleu_1)
    np.save("spf_std_bleu_3.npy", std_bleu_3)
