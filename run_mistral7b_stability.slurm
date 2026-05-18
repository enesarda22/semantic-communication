#!/bin/bash
#SBATCH --account=PAS2672
#SBATCH --job-name=mistral7b-stability
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=mistral7b_stability_%j.log
#SBATCH --mail-user=enesarda22@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

module load miniconda3/24.1.2-py310
conda activate semantic-communication

export HF_HOME=/fs/scratch/PAS2672/arda2/hf-cache
mkdir -p "$HF_HOME"

cd ~/repos/semantic-communication

MODEL=mistralai/Mistral-7B-Instruct-v0.3
ARGS_BASE="--pool frozen_pool_1500.jsonl --model $MODEL --dtype bf16"

echo "=== Prompt sweep (5 prompts, T=0, seed=42) ==="
python score_evaluator_oss.py $ARGS_BASE \
  --prompt-keys all --temperature 0.0 --seed 42 \
  --out mistral7b_prompts_1500.jsonl

echo "=== Temperature/repeats sweep ==="
python score_evaluator_oss.py $ARGS_BASE \
  --prompt-keys canonical --temperature 0.0 --seed 42 \
  --out mistral7b_T0.0_s42_1500.jsonl

for T in 0.3 0.7 1.0; do
  for SEED in 42 43 44; do
    echo "  T=$T seed=$SEED"
    python score_evaluator_oss.py $ARGS_BASE \
      --prompt-keys canonical --temperature $T --seed $SEED \
      --out mistral7b_T${T}_s${SEED}_1500.jsonl
  done
done
