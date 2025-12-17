## Installation

Transformers works with Python 3.11+, and [PyTorch](https://pytorch.org/get-started/locally/) 2.7+.

Create and activate a virtual environment with [conda](https://docs.conda.io/en/latest/).

```py
# conda
conda create -n transformers-lab python=3.11
conda activate transformers-lab

# install cuda-related package
conda install cuda-version=11.8 cudatoolkit=11.8.0 cudnn=8.9.7.29
conda install -c nvidia cuda-nvcc=11.8

# install torch-related package
pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 torchvision==0.22.1+cu118 --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu118
```

Install Transformers and timm in your virtual environment.

```py
cd transformers
# pip
pip install "transformers[torch]"
pip install -e ".[torch]"
pip install timm
```