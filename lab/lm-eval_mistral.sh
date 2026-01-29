#!/bin/bash

BASE_PATH="/home/jovyan/nas"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export HF_HUB_OFFLINE=1
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="${BASE_PATH}/yrc/.cache/huggingface"
export HF_TOKEN=""
export HF_HUB_ENABLE_HF_TRANSFER="1"
export NLTK_DATA="${BASE_PATH}/yrc/dataset/nltk_data"
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

# model configuration
MIXED_PRECISION="bf16"
DISCARD_RATE=0.1
DISCARD_BEFORE_LAYER="[true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]"
DISCARD_SEED=42
export RANDOM_DISCARD="{\"discard_rate\": ${DISCARD_RATE}, \"discard_before_layer\": ${DISCARD_BEFORE_LAYER}, \"discard_seed\": ${DISCARD_SEED}}"

# lm-eval configuration
PRETRAINED_MODEL_PATH="${BASE_PATH}/yrc/model/mistralai/Mistral-7B-Instruct-v0.3"
LOG_DIR=./workdir/lm-eval
ACCELERATE_CONFIG=./conf/accelerate_config_${MIXED_PRECISION}.yaml
case "$MIXED_PRECISION" in
    fp16)
        FULL_PRECISION="float16"
        ;;
    bf16)
        FULL_PRECISION="bfloat16"
        ;;
    fp32)
        FULL_PRECISION="float32"
        ;;
    *)
        FULL_PRECISION="unknown"
        ;;
esac
export LMEVAL_LOG_LEVEL="WARNING"

CONFIG_FILE_ARG="--config_file ${ACCELERATE_CONFIG}"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c)
            CONFIG_FILE_ARG="--config_file $2"
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

TASKS_STR="mmlu"
for seed in $(seq 1 2 21); do
    DISCARD_SEED=$seed
    for rate in $(seq 0 0.05 0.95); do
        DISCARD_RATE=$rate
        export RANDOM_DISCARD="{\"discard_rate\": ${DISCARD_RATE}, \"discard_before_layer\": ${DISCARD_BEFORE_LAYER}, \"discard_seed\": ${DISCARD_SEED}}"
        RUN_NAME=mistral_v0.3_7b_${MIXED_PRECISION}_discard-${DISCARD_RATE}_seed-${DISCARD_SEED}_layer-0

        echo "************************ ${RUN_NAME} ************************"
        if [ -d "$LOG_DIR/$RUN_NAME" ]; then
            echo "$LOG_DIR/$RUN_NAME already exists. Skipping..."
            continue
        fi

        accelerate launch $CONFIG_FILE_ARG --num_processes $NUM_DEVICES \
            -m lm_eval \
            --model hf \
            --model_args pretrained="${PRETRAINED_MODEL_PATH},dtype=${FULL_PRECISION},attn_implementation=flash_attention_2" \
            --tasks $TASKS_STR \
            --batch_size 1 \
            --log_samples \
            --output_path $LOG_DIR/$RUN_NAME
    done
done
