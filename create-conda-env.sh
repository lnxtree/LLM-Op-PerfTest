#!/bin/bash

conda create --name llm-op-perf python=3.11
source activate llm-op-perf

pip install packaging
pip install torch torchvision torchaudio
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git submodule update --init --recursive
python setup.py install