#!/bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH
export TORCH_CPP_LOG_LEVEL=ERROR
export CUDA_VISIBLE_DEVICES=0
NUM_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
BASE_PATH="/starry-nas"
TEACHER_MODEL_PATH='{"'"$BASE_PATH"'/yrc/model/transformers/vit_base_patch16_224.augreg2_in21k_ft_in1k/": "transformers"}'
STUDENT_MODEL_PATH="$BASE_PATH/yrc/model/yeruichen/vit-rd-base-patch16-224.augreg2_in21k_ft_in1k.sup-layerwise1/"
DATASET_PATH="$BASE_PATH/yrc/dataset/imagenet-1k/arrow/"
CKPT_SAVE_PATH="./ckpts/vit-rd-base-patch16-224.augreg2_in21k_ft_in1k.sup-layerwise1-discard-0.4-layer-0"

accelerate launch --num_processes=$NUM_DEVICES \
    src/vit_rd_distill_train.py \
    --teacher-model-path="${TEACHER_MODEL_PATH}" \
    --student-config-path="${STUDENT_MODEL_PATH}" \
    --dataset-path="${DATASET_PATH}" \
    --image-processor-path="" \
    --ckpt-save-path="${CKPT_SAVE_PATH}" \
    --ckpt-history-amount=0 \
    --log-level="warning" \
    --layer-num=12 \
    --learning-rate=1e-5 \
    --weight-decay=2e-2 \
    --warmup-ratio=0.06 \
    --num-epochs=50 \
    --earlystop-patience=0 \
    --discard-rate=0.4 \
    --discard-before-layers 0 \
    --temperature=5.0 \
    --alpha-param=0.2 \
    --beta-param=0.8
    # --resume-from-ckpt


