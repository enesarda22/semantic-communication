from semantic_communication.utils.general import plot
import numpy as np

separation_conventional_mean_bleu = np.load("Relay Final Results/conventional_mean_bleu_AWGN.npy")
ae_jscc_mean_bleu = np.load("Relay Final Results/ae_conventional_mean_bleu_AWGN.npy")
slf_mean_bleu = np.load("Relay Final Results/forward_AWGN_proposed_mean_bleu.npy")
spf_mean_bleu = np.load("Relay Final Results/predict_AWGN_proposed_mean_bleu.npy")
sentence_mean_bleu = np.load("Relay Final Results/sentence_AWGN_proposed_mean_bleu.npy")

distance_list = [1000, 1500, 1750, 2000, 2500, 3000]
gamma_list = [0.1, 0.3, 0.5, 0.7, 0.9]

plot(d_sd_list=distance_list, y_label="BLEU Score", gamma_list=gamma_list, separation_conventional=separation_conventional_mean_bleu,
     SPF=spf_mean_bleu, SLF=slf_mean_bleu, sentence_decode=sentence_mean_bleu,
         sentence_predict=None, AE_baseline=ae_jscc_mean_bleu, save=True, show=False)

separation_conventional_mean_sbert = np.load("Relay Final Results/conventional_mean_sbert_semantic_sim_AWGN.npy")
ae_jscc_mean_sbert = np.load("Relay Final Results/ae_conventional_mean_sbert_semantic_sim_AWGN.npy")
slf_mean_sbert = np.load("Relay Final Results/forward_AWGN_proposed_mean_sbert_semantic_sim.npy")
spf_mean_sbert = np.load("Relay Final Results/predict_AWGN_proposed_mean_sbert_semantic_sim.npy")
sentence_mean_sbert = np.load("Relay Final Results/sentence_AWGN_proposed_mean_sbert_semantic_sim.npy")

plot(d_sd_list=distance_list, y_label="SBERT Semantic Similarity", gamma_list=gamma_list, separation_conventional=separation_conventional_mean_sbert,
     SPF=spf_mean_sbert, SLF=slf_mean_sbert, sentence_decode=sentence_mean_sbert,
         sentence_predict=None, AE_baseline=ae_jscc_mean_sbert, save=True, show=False)


separation_conventional_mean_gpt = np.load("Relay Final Results/conventional_mean_semantic_sim_AWGN.npy")
ae_jscc_mean_gpt = np.load("Relay Final Results/ae_conventional_mean_semantic_sim_AWGN.npy")
slf_mean_gpt = np.load("Relay Final Results/forward__AWGNproposed_mean_semantic_sim.npy")
spf_mean_gpt = np.load("Relay Final Results/predict__AWGNproposed_mean_semantic_sim.npy")
sentence_mean_gpt = np.load("Relay Final Results/sentence__AWGNproposed_mean_semantic_sim.npy")

plot(d_sd_list=distance_list, y_label="GPT Semantic Similarity", gamma_list=gamma_list, separation_conventional=separation_conventional_mean_gpt,
     SPF=spf_mean_gpt, SLF=slf_mean_gpt, sentence_decode=sentence_mean_gpt,
         sentence_predict=None, AE_baseline=ae_jscc_mean_gpt, save=True, show=False)
