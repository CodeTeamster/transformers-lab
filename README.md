## Installation

Transformers works with Python 3.11+, and [PyTorch](https://pytorch.org/get-started/locally/) 2.7+.

Create and activate a virtual environment with [conda](https://docs.conda.io/en/latest/).

```py
# conda
conda create -n transformers-lab python=3.11
conda activate transformers-lab

# install cuda-related package
conda install cuda-version=12.1 cudnn=8.9.7.29
conda install -c nvidia cuda-nvcc=12.1

# install torch-related package
pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu121
```

Install Transformers and timm in your virtual environment.

```py
cd transformers
# pip
pip install -e ".[torch]"
pip install timm
pip install scikit-learn

conda install openjdk=8
cd ../lmms-eval
# pip
pip install -e .
```

Install FlashAttention2 in your virtual environment if necessary.

```py
# pip
pip install packaging
pip install ninja
pip install flash-attn==2.7.4.post1 --no-build-isolation
```