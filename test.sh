#!/bin/bash
#SBATCH --job-name=flash_attn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --output=slurm-output/slurm-%j.out
#SBATCH --exclude=hepnode0

nvcc --version
nvidia-smi

python scripts/test-flash-attn.py --seq_length 2048 --num_kv_heads 4
python scripts/test-flash-attn.py --seq_length 4096 --num_kv_heads 4
python scripts/test-flash-attn.py --seq_length 8192 --num_kv_heads 4
python scripts/test-flash-attn.py --seq_length 16384 --num_kv_heads 4
python scripts/test-flash-attn.py --seq_length 32768 --num_kv_heads 4
