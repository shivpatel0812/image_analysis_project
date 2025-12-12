#!/bin/bash
#SBATCH --job-name=mae2d_pretrain
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --output=mae2d_pretrain_%j.out

export USER=njt4xc
source ~/venv_selfmedmae/bin/activate
cd ~/SelfMedMAE

echo "Starting 2D MAE Pre-training"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python main.py configs/mae2d_btcv_1gpu.yaml

echo "Training completed at $(date)"

