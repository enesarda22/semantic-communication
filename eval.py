import argparse
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI

from nltk.translate.bleu_score import sentence_bleu

from semantic_communication.models.semantic_transformer import SemanticTransformer
from semantic_communication.models.transceiver import (
    Transceiver,
    ChannelEncoder,
    ChannelDecoder,
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


def plotter(x_axis_values, y_axis_values, x_label, y_label, title):
    plt.figure()
    plt.plot(x_axis_values, y_axis_values)
    plt.grid()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(f"{title}.png", dpi=400)


def semantic_similarity_score(target_sentences, received_sentences):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You are skilled in evaluating how similar the two sentences are. Provide a number between -1 "
                        "and 1 denoting the semantic similarity score for given sentences A and B with precision "
                        "0.01. 1 means they are perfectly similar and -1 means they are opposite while 0 means their "
                        "meanings are uncorrelated."},
            {"role": "user", "content": f"A=({target_sentences})  B=({received_sentences})"}
        ]
    )
    if completion.choices[0].finish_reason == "stop":
        return float(completion.choices[0].message.content)
    else:
        raise ValueError("Finish reason is not stop.")


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
    parser.add_argument("--API-KEY", type=str)  # API KEY

    add_semantic_decoder_args(parser)
    add_channel_model_args(parser)
    add_data_args(parser)

    # test args
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--gamma-list", nargs="+", type=float)
    parser.add_argument("--d-list", nargs="+", type=float)
    parser.add_argument("--n-test", default=50, type=int)

    args = parser.parse_args()
    device = get_device()
    set_seed()

    client = OpenAI(api_key=args.API_KEY)

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

    relay_semantic_encoder = SemanticEncoder(
        label_encoder=data_handler.label_encoder,
        max_length=args.max_length,
        mode=args.mode if args.mode == "sentence" else "forward",
        rate=1 if args.mode == "sentence" else None,
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
        bert=relay_semantic_encoder.bert,
        pad_idx=data_handler.label_encoder.pad_id,
    ).to(device)

    transceiver = Transceiver(
        src_relay_transformer=semantic_transformer,
        relay_semantic_encoder=relay_semantic_encoder,
        relay_channel_encoder=relay_channel_encoder,
        dst_channel_decoder=dst_channel_decoder,
        dst_semantic_decoder=dst_semantic_decoder,
        channel=channel,
        max_length=args.max_length,
    ).to(device)
    load_model(transceiver, args.transceiver_path)

    transceiver.eval()

    n_d = len(args.d_list)
    n_gamma = len(args.gamma_list)

    mean_semantic_sim = np.zeros((n_d, n_gamma))
    mean_bleu_1 = np.zeros((n_d, n_gamma))
    mean_bleu_3 = np.zeros((n_d, n_gamma))

    std_semantic_sim = np.zeros((n_d, n_gamma))
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

                predicted_ids, probs = transceiver.generate(
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
                    cosine_scores.append(semantic_similarity_score(s1, s2))
                    bleu1_scores.append(bleu_1gram(s1, s2))
                    bleu3_scores.append(bleu_3gram(s1, s2))

                if len(cosine_scores) > args.n_test:
                    break

            n_test_samples = len(cosine_scores)
            mean_semantic_sim[distance_index, gamma_index] = np.mean(cosine_scores)
            mean_bleu_1[distance_index, gamma_index] = np.mean(bleu1_scores)
            mean_bleu_3[distance_index, gamma_index] = np.mean(bleu3_scores)

            std_semantic_sim[distance_index, gamma_index] = np.std(cosine_scores, ddof=1) / np.sqrt(n_test_samples)
            std_bleu_1[distance_index, gamma_index] = np.std(bleu1_scores, ddof=1) / np.sqrt(n_test_samples)
            std_bleu_3[distance_index, gamma_index] = np.std(bleu3_scores, ddof=1) / np.sqrt(n_test_samples)

            np.save("proposed_mean_semantic_sim.npy", mean_semantic_sim)
            np.save("proposed_mean_bleu_1.npy", mean_bleu_1)
            np.save("proposed_mean_bleu_3.npy", mean_bleu_3)

            np.save("proposed_std_semantic_sim.npy", std_semantic_sim)
            np.save("proposed_std_bleu_1.npy", std_bleu_1)
            np.save("proposed_std_bleu_3.npy", std_bleu_3)
