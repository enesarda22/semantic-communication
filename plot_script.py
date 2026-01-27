from semantic_communication.utils.general import plot_three_metrics
import numpy as np

distance_list = [1000, 1500, 1750, 2000, 2500, 3000]
gamma_list = [0.1, 0.3, 0.5, 0.7, 0.9]

separation_conventional_mean_bleu = np.load("Relay Final Results/conventional_mean_bleu_AWGN.npy")
ae_jscc_mean_bleu = np.load("Relay Final Results/ae_conventional_mean_bleu_AWGN.npy")
slf_mean_bleu = np.load("Relay Final Results/forward_AWGN_proposed_mean_bleu.npy")
spf_mean_bleu = np.load("Relay Final Results/predict_AWGN_proposed_mean_bleu.npy")
sentence_mean_bleu = np.load("Relay Final Results/sentence_AWGN_proposed_mean_bleu.npy")
llm_baseline_mean_bleu = np.load("Relay Final Results/llmsc_mean_bleu_AWGN.npy")

separation_conventional_mean_sbert = np.load("Relay Final Results/conventional_mean_sbert_semantic_sim_AWGN.npy")
ae_jscc_mean_sbert = np.load("Relay Final Results/ae_conventional_mean_sbert_semantic_sim_AWGN.npy")
slf_mean_sbert = np.load("Relay Final Results/forward_AWGN_proposed_mean_sbert_semantic_sim.npy")
spf_mean_sbert = np.load("Relay Final Results/predict_AWGN_proposed_mean_sbert_semantic_sim.npy")
sentence_mean_sbert = np.load("Relay Final Results/sentence_AWGN_proposed_mean_sbert_semantic_sim.npy")
llm_baseline_mean_sbert = np.load("Relay Final Results/llmsc_mean_sbert_semantic_sim_AWGN.npy")

separation_conventional_mean_gpt = np.load("Relay Final Results/conventional_mean_semantic_sim_AWGN.npy")
ae_jscc_mean_gpt = np.load("Relay Final Results/ae_conventional_mean_semantic_sim_AWGN.npy")
slf_mean_gpt = np.load("Relay Final Results/forward__AWGNproposed_mean_semantic_sim.npy")
spf_mean_gpt = np.load("Relay Final Results/predict__AWGNproposed_mean_semantic_sim.npy")
sentence_mean_gpt = np.load("Relay Final Results/sentence__AWGNproposed_mean_semantic_sim.npy")
llm_baseline_mean_gpt = np.load("Relay Final Results/llmsc_mean_semantic_sim_AWGN.npy")

plot_three_metrics(
    d_sd_list=distance_list,
    gamma_list=gamma_list,

    bleu_sep=separation_conventional_mean_bleu,
    bleu_spf=spf_mean_bleu,
    bleu_slf=slf_mean_bleu,
    bleu_ssf=sentence_mean_bleu,
    bleu_ae=ae_jscc_mean_bleu,
    bleu_llm=llm_baseline_mean_bleu,

    sbert_sep=separation_conventional_mean_sbert,
    sbert_spf=spf_mean_sbert,
    sbert_slf=slf_mean_sbert,
    sbert_ssf=sentence_mean_sbert,
    sbert_ae=ae_jscc_mean_sbert,
    sbert_llm=llm_baseline_mean_sbert,

    gpt_sep=separation_conventional_mean_gpt,
    gpt_spf=spf_mean_gpt,
    gpt_slf=slf_mean_gpt,
    gpt_ssf=sentence_mean_gpt,
    gpt_ae=ae_jscc_mean_gpt,
    gpt_llm=llm_baseline_mean_gpt,

    save_dir="Plots",
    show=False,
)
