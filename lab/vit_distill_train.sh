#!/bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH
export TORCH_CPP_LOG_LEVEL=ERROR
export CUDA_VISIBLE_DEVICES=0
NUM_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
TEACHERS_MODEL_PATH='{
    "/home/jovyan/nas/yrc/model/timm/vit_base_patch16_224.augreg_in1k/model.safetensors": "timm",
    "/home/jovyan/nas/yrc/model/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k/model.safetensors": "timm",
    "/home/jovyan/nas/yrc/model/timm/deit_base_patch16_224.fb_in1k/model.safetensors": "timm",
    "/home/jovyan/nas/yrc/model/timm/deit3_base_patch16_224.fb_in22k_ft_in1k/model.safetensors": "timm"
}'
TEACHERS_MODEL_PATH='{
    "/home/jovyan/nas/yrc/model/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k/model.safetensors": "timm",
    "/home/jovyan/nas/yrc/model/timm/deit3_base_patch16_224.fb_in22k_ft_in1k/model.safetensors": "timm"
}'
TEACHERS_MODEL_PATH=$(echo "$TEACHERS_MODEL_PATH" | tr -d '\n' | sed 's/,}/}/')
STUDENT_MODEL_PATH="/home/jovyan/nas/yrc/model/google/vit-base-patch16-224/"

accelerate launch --num_processes=$NUM_DEVICES \
    src/vit_distill_train.py \
    --teachers-model-path="${TEACHERS_MODEL_PATH}" \
    --student-model-path="${STUDENT_MODEL_PATH}" \
    --dataset-path="/home/jovyan/nas/yrc/dataset/tiny-imagenet/arrow/" \
    --image-processor-path="" \
    --ckpt-save-path="/home/jovyan/nas/yrc/workspace/hugging-face-experiments/experiments/checkpoints/vit-base-patch16-224-mt2-distill" \
    --log-level="warning" \
    --num-epochs=300 \
    --earlystop-patience=10
    # --resume-from-ckpt \