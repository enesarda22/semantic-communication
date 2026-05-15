#!/bin/bash
#SBATCH --account=PAS2672
#SBATCH --job-name=mixtral8x7b-1500
#SBATCH --gpus-per-node=4
#SBATCH --time=08:00:00
#SBATCH --output=mixtral8x7b_1500_%j.log
#SBATCH --mail-user=enesarda22@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

module load miniconda3/24.1.2-py310
conda activate semantic-communication

# Use scratch for HF cache to avoid home-dir quota issues
export HF_HOME=/fs/scratch/PAS2672/arda2/hf-cache
mkdir -p "$HF_HOME"

cd ~/repos/semantic-communication

python score_evaluator_oss.py \
  --pool frozen_pool_1500.jsonl \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --quant 8bit \
  --dtype bf16 \
  --out mixtral8x7b_1500.jsonl
