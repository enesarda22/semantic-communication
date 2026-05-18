#!/bin/bash
#SBATCH --account=PAS2672
#SBATCH --job-name=mixtral-temps
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=mixtral8x7b_temps_%j.log
#SBATCH --mail-user=enesarda22@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

module load miniconda3/24.1.2-py310
conda activate semantic-communication

export HF_HOME=/fs/scratch/PAS2672/arda2/hf-cache
mkdir -p "$HF_HOME"

cd ~/repos/semantic-communication

MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1
ARGS_BASE="--pool frozen_pool_1500.jsonl --model $MODEL --quant 8bit --dtype bf16"

# T=0 (1 seed, deterministic baseline)
echo "=== T=0 (canonical, deterministic) ==="
python score_evaluator_oss.py $ARGS_BASE \
  --prompt-keys canonical --temperature 0.0 --seed 42 \
  --out mixtral8x7b_T0.0_s42_1500.jsonl

# T > 0 sweep with 3 seeds each
for T in 0.3 0.7 1.0; do
  for SEED in 42 43 44; do
    echo "  T=$T seed=$SEED"
    python score_evaluator_oss.py $ARGS_BASE \
      --prompt-keys canonical --temperature $T --seed $SEED \
      --out mixtral8x7b_T${T}_s${SEED}_1500.jsonl
  done
done
