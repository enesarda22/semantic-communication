import argparse
import numpy as np
import torch
from openai import OpenAI
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
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
from sentence_transformers import SentenceTransformer
from semantic_communication.utils.eval_functions import *
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model args
    parser.add_argument("--transceiver-path", type=str)
    parser.add_argument("--API-KEY", type=str)  # API KEY

    add_semantic_decoder_args(parser)
    add_channel_model_args(parser)
    add_data_args(parser)

    # test args
    parser.add_argument("--batch-size", default=125, type=int)
    parser.add_argument("--gamma-list", nargs="+", type=float)
    parser.add_argument("--d-list", nargs="+", type=float)
    parser.add_argument("--n-test", default=500, type=int)

    args = parser.parse_args()
    device = get_device()
    set_seed()

    results_dir = "Results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    client = OpenAI(api_key=args.API_KEY)
    sbert_eval_model = SentenceTransformer("all-MiniLM-L6-v2")

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
    mean_sbert_semantic_sim = np.zeros((n_d, n_gamma))
    mean_bleu_1 = np.zeros((n_d, n_gamma))
    mean_bleu = np.zeros((n_d, n_gamma))

    std_semantic_sim = np.zeros((n_d, n_gamma))
    std_sbert_semantic_sim = np.zeros((n_d, n_gamma))
    std_bleu_1 = np.zeros((n_d, n_gamma))
    std_bleu = np.zeros((n_d, n_gamma))
    smoothing_function = SmoothingFunction().method1

    records = []
    # for each d_sd
    for distance_index, d_sd in enumerate(args.d_list):
        # for each gamma in gamma list
        for gamma_index, gamma in enumerate(args.gamma_list):
            print(f"Simulating for distance: {d_sd}  - Gamma: {gamma}")

            sbert_semantic_sim_scores = []
            cosine_scores = []
            bleu1_scores = []
            bleu_scores = []

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

                # find the end of sentences
                sep_indices = torch.argmax((predicted_ids == 2).long(), dim=1)
                input_ids_list = []
                for i in range(predicted_ids.shape[0]):
                    k = sep_indices[i]
                    if k == 0:  # no [SEP] predicted
                        input_ids_list.append(predicted_ids[i, :])
                    else:
                        input_ids_list.append(predicted_ids[i, : k + 1])

                token_ids_list = [
                    semantic_encoder.label_encoder.inverse_transform(input_ids)
                    for input_ids in input_ids_list
                ]

                predicted_tokens = semantic_encoder.get_tokens(
                    token_ids=token_ids_list,
                    skip_special_tokens=True,
                )

                input_tokens = semantic_encoder.get_tokens(
                    ids=encoder_idx,
                    skip_special_tokens=True,
                )

                for s1, s2 in zip(input_tokens, predicted_tokens):
                    # print(f"True Sentence: {s1}\nPredicted Sentence: {s2}\n")
                    # sim_score = semantic_similarity_score(s1, s2, client)
                    bleu_1_score = sentence_bleu([word_tokenize(s1)], word_tokenize(s2), weights=[1, 0, 0, 0], smoothing_function=smoothing_function)
                    bleu_score = sentence_bleu([word_tokenize(s1)], word_tokenize(s2), smoothing_function=smoothing_function)
                    sbert_sim_score = sbert_semantic_similarity_score(s1, s2, sbert_model=sbert_eval_model)

                    # cosine_scores.append(sim_score)
                    bleu1_scores.append(bleu_1_score)
                    bleu_scores.append(bleu_score)
                    sbert_semantic_sim_scores.append(sbert_sim_score)
                    # records.append([d_sd, gamma, s1, s2, sim_score, bleu_1_score, bleu_score, sbert_sim_score])

                if len(bleu1_scores) >= args.n_test:
                    break

            n_test_samples = len(bleu1_scores)
            cosine_scores = [x for x in cosine_scores if not np.isnan(x)]

            # mean_semantic_sim[distance_index, gamma_index] = np.mean(cosine_scores)
            mean_sbert_semantic_sim[distance_index, gamma_index] = np.mean(sbert_semantic_sim_scores)
            mean_bleu_1[distance_index, gamma_index] = np.mean(bleu1_scores)
            mean_bleu[distance_index, gamma_index] = np.mean(bleu_scores)
            #
            # std_semantic_sim[distance_index, gamma_index] = np.std(
            #     cosine_scores, ddof=1
            # ) / np.sqrt(n_test_samples)
            std_bleu_1[distance_index, gamma_index] = np.std(
                bleu1_scores, ddof=1
            ) / np.sqrt(n_test_samples)
            std_bleu[distance_index, gamma_index] = np.std(
                bleu_scores, ddof=1
            ) / np.sqrt(n_test_samples)

            std_sbert_semantic_sim[distance_index, gamma_index] = np.std(sbert_semantic_sim_scores, ddof=1) / np.sqrt(n_test_samples)

            # np.save(os.path.join(results_dir, f"{args.mode}_proposed_mean_semantic_sim.npy"), mean_semantic_sim)
            np.save(os.path.join(results_dir, f"{args.mode}_proposed_mean_sbert_semantic_sim.npy"), mean_sbert_semantic_sim)
            np.save(os.path.join(results_dir, f"{args.mode}_proposed_mean_bleu_1.npy"), mean_bleu_1)
            np.save(os.path.join(results_dir, f"{args.mode}_proposed_mean_bleu.npy"), mean_bleu)

            # np.save(os.path.join(results_dir, f"{args.mode}_proposed_std_semantic_sim.npy"), std_semantic_sim)
            np.save(os.path.join(results_dir, f"{args.mode}_proposed_std_sbert_semantic_sim.npy"), std_sbert_semantic_sim)
            np.save(os.path.join(results_dir, f"{args.mode}_proposed_std_bleu_1.npy"), std_bleu_1)
            np.save(os.path.join(results_dir, f"{args.mode}_proposed_std_bleu.npy"), std_bleu)

            # df = pd.DataFrame(records, columns=['d_sd', "Gamma", 'Sentence 1', 'Sentence 2', 'Semantic Similarity Score', 'BLEU 1 Gram Score', 'BLEU Score', "SBERT Semantic Score"])
            # df.to_excel(os.path.join(results_dir, f'{args.mode}_proposed_output.xlsx'), index=False)
