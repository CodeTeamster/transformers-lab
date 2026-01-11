#!/bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH
export TORCH_CPP_LOG_LEVEL=ERROR
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
BASE_PATH="/home/jovyan/nas"
TEACHER_MODEL_PATH='{"'"$BASE_PATH"'/yrc/model/yeruichen/vit-rd-base-patch16-224.sup-0/": "transformers"}'
STUDENT_MODEL_PATH="$BASE_PATH/yrc/model/yeruichen/vit-rd-base-patch16-224.sup-12/"
DATASET_PATH="$BASE_PATH/yrc/dataset/imagenet-1k/arrow/"
CKPT_SAVE_PATH="./ckpts/vit.sup-12-discard-ud0.3-layer-all"

accelerate launch --num_processes=$NUM_DEVICES \
    src/vit_rd_distill_train.py \
    --teacher-model-path="${TEACHER_MODEL_PATH}" \
    --student-config-path="${STUDENT_MODEL_PATH}" \
    --dataset-path="${DATASET_PATH}" \
    --image-processor-path="" \
    --ckpt-save-path="${CKPT_SAVE_PATH}" \
    --ckpt-history-amount=5 \
    --log-level="warning" \
    --layer-num=12 \
    --learning-rate=2e-5 \
    --weight-decay=1e-2 \
    --warmup-ratio=0.06 \
    --num-epochs=50 \
    --earlystop-patience=0 \
    --discard-rate=-0.3 \
    --discard-before-layers 0 1 2 3 4 5 6 7 8 9 10 11 \
    --temperature=5.0 \
    --alpha-param=0.2 \
    --beta-param=0.8
    # --resume-from-ckpt


