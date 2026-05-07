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

### Training
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

### Round-2 evaluator-comparison experiments (JSTSP revision)

Scripts that read precomputed (source, reconstruction) pairs from the
artifacts/ xlsx dumps and score them with multiple LLM-based and
embedding-based evaluators. They do **not** load any trained transceiver and
have no GPU dependency on their own (`score_evaluator_oss.py` does load
HuggingFace LLMs, the others are CPU-only). Outputs feed into the expanded
Table III and the new evaluator-comparison subsubsection in v13.

```
# 0. Build the 600-pair frozen pool (5 methods x 3 distances x 40 sentences)
python build_frozen_pool.py \
  --artifacts-dir /path/to/artifacts \
  --out frozen_pool.jsonl

# 1. A0 pilot - validate JSON parsing, retries, model availability
python score_evaluator_openai.py --pool frozen_pool.jsonl --pilot 10 \
  --models gpt-4o-mini-YYYY-MM-DD --out pilot_openai.jsonl
python score_evaluator_oss.py --pool frozen_pool.jsonl --pilot 10 \
  --model meta-llama/Llama-3.1-8B-Instruct --out pilot_llama.jsonl
python score_evaluator_bertscore.py --pool frozen_pool.jsonl --pilot 10 \
  --out pilot_bertscore.jsonl

# 2. Full GPT stability (3 dated GPT models x 5 prompts x 600 pairs)
python score_evaluator_openai.py --pool frozen_pool.jsonl --prompt-keys all \
  --models gpt-4o-mini-YYYY-MM-DD,gpt-4.1-nano-YYYY-MM-DD,gpt-5-nano-YYYY-MM-DD \
  --out gpt_stability_raw.jsonl

# 3. Determinism check (50-pair subset x 3 calls at temp 0)
python repeat_check.py --pool frozen_pool.jsonl --subset 50 --repeats 3 \
  --model gpt-4o-mini-YYYY-MM-DD --out repeat_check.json

# 4. Non-GPT comparison
python score_evaluator_oss.py --pool frozen_pool.jsonl \
  --model meta-llama/Llama-3.1-8B-Instruct --out llama8b.jsonl
python score_evaluator_oss.py --pool frozen_pool.jsonl \
  --model meta-llama/Llama-3.3-70B-Instruct --quant 4bit --out llama70b.jsonl
python score_evaluator_oss.py --pool frozen_pool.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct --out qwen7b.jsonl
python score_evaluator_oss.py --pool frozen_pool.jsonl \
  --model mistralai/Mistral-7B-Instruct-v0.3 --out mistral7b.jsonl
python score_evaluator_bertscore.py --pool frozen_pool.jsonl \
  --out bertscore_raw.jsonl

# 5. Aggregate everything
python aggregate_evaluators.py \
  --gpt gpt_stability_raw.jsonl \
  --oss llama8b.jsonl,llama70b.jsonl,qwen7b.jsonl,mistral7b.jsonl \
  --bertscore bertscore_raw.jsonl \
  --out summary/
```

Prompt definitions live in `prompts.json` (canonical + 5 paraphrased variants
+ repair). Set `OPENAI_API_KEY` in env or pass `--api-key` to the OpenAI
scripts.

