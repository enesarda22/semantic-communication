# Semantic Forwarding for Next Generation Relay Networks

This repository contains the code for simulations in the paper:

Enes Arda, Emrecan Kutay and Aylin Yener, Semantic Forwarding for Next Generation Relay Networks, in 2024 58th Annual Conference on Information Sciences and Systems, CISS’24, Princeton, NJ, USA, Mar. 2024.

Please cite this paper if you use or refer to this code:
```bibtex
@inproceedings{arda_semantic_2024,
	title = {Semantic {Forwarding} for {Next} {Generation} {Relay} {Networks}},
	url = {https://ieeexplore.ieee.org/document/10480169/?arnumber=10480169},
	doi = {10.1109/CISS59072.2024.10480169},
	abstract = {We consider cooperative semantic text communications facilitated by a relay node. We propose two types of semantic forwarding: semantic lossy forwarding (SLF) and semantic predict-and-forward (SPF). Both are machine learning aided approaches, and, in particular, utilize attention mechanisms at the relay to establish a dynamic semantic state, updated upon receiving a new source signal. In the SLF model, the semantic state is used to decode the received source signal; whereas in the SPF model, it is used to predict the next source signal, enabling proactive forwarding. Our proposed forwarding schemes do not need any channel state information and exhibit consistent performance regardless of the relay’s position. Our results demonstrate that the proposed semantic forwarding techniques outperform conventional semantic-agnostic baselines.},
	urldate = {2024-11-02},
	booktitle = {2024 58th {Annual} {Conference} on {Information} {Sciences} and {Systems} ({CISS})},
	author = {Arda, Enes and Kutay, Emrecan and Yener, Aylin},
	month = mar,
	year = {2024},
	note = {ISSN: 2837-178X},
	keywords = {6G, Cooperative communication, Data mining, Machine learning, Predictive models, relay network, Relay networks, Semantic communications, semantic lossy forwarding, semantic predict-and-forward, Semantics, Simulation},
	pages = {1--6},
}
```

### Training Commands
```
python train_semantic_transformer.py \
--data-fp ~/data \
--checkpoint-path ~/data/checkpoints/improved-semantic-transformer-with-channel \
--mode sentence \
--rate 1 \
--batch-size 512 \
--n-epochs 30 \
--lr 6e-4 \
--n-blocks 6 \
--n-heads 6 \
--channel-block-input-dim 384 \
--channel-block-latent-dim 96 \
--channel-type AWGN \
--alpha 4 \
--sig-pow 1 \
--noise-pow 4e-15 \
--d-min 1e3 \
--d-max 3e3 \
--gamma-min 0.1 \
--gamma-max 0.9
```

```
torchrun --standalone --nproc_per_node=4 train_end_to_end.py \
--data-fp ~/data \
--checkpoint-path ~/data/checkpoints/ \
--semantic-transformer-path ~/data/checkpoints/improved-semantic-transformer-with-channel/semantic-transformer/semantic_transformer_30.pt \
--mode sentence \
--rate 1 \
--batch-size 1024 \
--n-epochs 15 \
--eval-iter 400 \
--lr 1e-3 \
--n-blocks 6 \
--n-heads 6 \
--channel-block-input-dim 384 \
--channel-block-latent-dim 96 \
--channel-type AWGN \
--alpha 4 \
--sig-pow 1 \
--noise-pow 4e-15 \
--d-min 1e3 \
--d-max 3e3 \
--gamma-min 0.1 \
--gamma-max 0.9
```

```
python baseline_train_source_relay.py \
--data-fp ~/data \
--checkpoint-path ~/data/checkpoints/ \
--batch-size 512 \
--n-epochs 30 \
--lr 5e-4 \
--channel-block-input-dim 384 \
--channel-block-latent-dim 4 \
--channel-type AWGN \
--alpha 4 \
--sig-pow 1 \
--noise-pow 4e-15 \
--d-min 1e3 \
--d-max 3e3 \
--gamma-min 0.1 \
--gamma-max 0.9
```

```
python baseline_train_entire_network.py \
--data-fp ~/data \
--checkpoint-path ~/data/checkpoints/ \
--baseline-tx-relay-path ~/data/checkpoints/baseline-tx-relay/baseline_tx_relay_24.pt \
--batch-size 512 \
--n-epochs 15 \
--lr 5e-4 \
--channel-block-input-dim 384 \
--channel-block-latent-dim 4 \
--channel-type AWGN \
--alpha 4 \
--sig-pow 1 \
--noise-pow 4e-15 \
--d-min 1e3 \
--d-max 3e3 \
--gamma-min 0.1 \
--gamma-max 0.9
```