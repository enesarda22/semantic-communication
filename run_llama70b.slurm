#!/bin/bash
#SBATCH --account=PAS2672
#SBATCH --job-name=llama70b-eval
#SBATCH --gpus-per-node=4
#SBATCH --time=03:00:00
#SBATCH --output=llama70b_%j.log

module load miniconda3/24.1.2-py310
conda activate semantic-communication

cd ~/repos/semantic-communication

python score_evaluator_oss.py \
  --pool frozen_pool.jsonl \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --quant 4bit \
  --out llama70b.jsonl
