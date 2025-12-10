#!/bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH
export TORCH_CPP_LOG_LEVEL=ERROR
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
TEACHER_MODEL_PATH='{"/home/jovyan/nas/yrc/model/transformers/vit_base_patch16_224.augreg2_in21k_ft_in1k/": "transformers"}'
STUDENT_MODEL_PATH="/home/jovyan/nas/yrc/model/yeruichen/vit-graft-base-patch16-224.augreg2_in21k_ft_in1k.sup-6/"
DATASET_PATH="/home/jovyan/nas/yrc/dataset/imagenet-1k/arrow/"
CKPT_SAVE_PATH="/home/jovyan/nas/yrc/workspace/hugging-face-experiments/experiments/checkpoints/vit-graft-base-patch16-224-augreg2_in21k_ft_in1k-sup-6-discard-0.2-layer-all"

accelerate launch --num_processes=$NUM_DEVICES \
    src/graft_distill_train.py \
    --teacher-model-path="${TEACHER_MODEL_PATH}" \
    --student-model-path="${STUDENT_MODEL_PATH}" \
    --dataset-path="${DATASET_PATH}" \
    --image-processor-path="" \
    --ckpt-save-path="${CKPT_SAVE_PATH}" \
    --ckpt-history-amount=3 \
    --log-level="warning" \
    --layer-num=12 \
    --learning-rate=1e-5 \
    --weight-decay=2e-2 \
    --warmup-ratio=0.06 \
    --num-epochs=50 \
    --earlystop-patience=3 \
    --discard-rate=0.2 \
    --discard-before-layers 0 1 2 3 4 5 6 7 8 9 10 11 \
    --temperature=5.0 \
    --alpha-param=0.2 \
    --beta-param=0.8
    # --resume-from-ckpt


