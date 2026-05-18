#!/bin/bash
#SBATCH --account=PAS2672
#SBATCH --job-name=mixtral-prompts
#SBATCH --gpus-per-node=4
#SBATCH --time=18:00:00
#SBATCH --output=mixtral8x7b_prompts_%j.log
#SBATCH --mail-user=enesarda22@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

module load miniconda3/24.1.2-py310
conda activate semantic-communication

export HF_HOME=/fs/scratch/PAS2672/arda2/hf-cache
mkdir -p "$HF_HOME"

cd ~/repos/semantic-communication

MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1
ARGS_BASE="--pool frozen_pool_1500.jsonl --model $MODEL --quant 8bit --dtype bf16"

echo "=== Prompt sweep (5 prompts, T=0, seed=42) ==="
python score_evaluator_oss.py $ARGS_BASE \
  --prompt-keys all --temperature 0.0 --seed 42 \
  --out mixtral8x7b_prompts_1500.jsonl
