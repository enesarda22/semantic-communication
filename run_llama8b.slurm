#!/bin/bash
#SBATCH --account=PAS2672
#SBATCH --job-name=llama8b-1500
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=llama8b_1500_%j.log
#SBATCH --mail-user=enesarda22@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

module load miniconda3/24.1.2-py310
conda activate semantic-communication

cd ~/repos/semantic-communication

python score_evaluator_oss.py \
  --pool frozen_pool_1500.jsonl \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dtype bf16 \
  --out llama8b_1500.jsonl
