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
# pytest   -v flash-attention/tests/test_flash_attn.py
python scripts/test-flash-attn.py --seq_length 4096