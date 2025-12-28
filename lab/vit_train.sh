#!/bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH
export TORCH_CPP_LOG_LEVEL=ERROR
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
BASE_PATH="/home/jovyan/nas"
PRETRAINED_MODEL_PATH="$BASE_PATH/yrc/model/yeruichen/vit-rd-base-patch16-224.augreg2_in21k_ft_in1k.sup-0/"
DATASET_PATH="$BASE_PATH/yrc/dataset/imagenet-1k/arrow/"
CKPT_SAVE_PATH="./ckpts/evit-discard-0.3-layer369"

accelerate launch --num_processes=$NUM_DEVICES \
    src/transformers_vit_train.py \
    --pretrained-model-path="${PRETRAINED_MODEL_PATH}" \
    --dataset-path="${DATASET_PATH}" \
    --image-processor-path="" \
    --ckpt-save-path="${CKPT_SAVE_PATH}" \
    --ckpt-history-amount=6 \
    --log-level="warning" \
    --learning-rate=1e-5 \
    --weight-decay=2e-2 \
    --warmup-ratio=0.06 \
    --num-epochs=50 \
    --earlystop-patience=0 \
    --discard-rate=0.3
    # --resume-from-ckpt


