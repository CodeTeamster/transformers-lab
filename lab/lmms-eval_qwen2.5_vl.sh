#!/bin/bash

# max_pixels=12845056
BASE_PATH="/home/jovyan/nas"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="${BASE_PATH}/yrc/.cache/huggingface"
export HF_TOKEN=""
export HF_HUB_ENABLE_HF_TRANSFER="1"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NLTK_DATA="${BASE_PATH}/yrc/dataset/nltk_data"

NUM_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
PRETRAINED_MODEL_PATH="${BASE_PATH}/yrc/model/Qwen/Qwen2.5-VL-3B-Instruct"
LOG_DIR=./workdir/lmmseval
RUN_NAME=qwen_2.5_vl_3b

CONFIG_FILE_ARG=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config_file)
            CONFIG_FILE_ARG="--config_file $2"
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

# 1) Comprehensive Evaluation: MME and MMBench
# 2) Mathematical Reasoning: MathVista and Math-Vision
# 3) Optical Character Recognition: SEEDBench-2-Plus and OCRBench
# 4) Instruction Following: MIA-Bench
# 5) Multidisciplinary Knowledge: ScienceQA
# 6) Hallucination: POPE and HallusionBench
TASKS_STR="mme,mmbench,mathvision_test,mathvista,seedbench_2_plus,ocrbench,mia_bench,scienceqa,pope,hallusion_bench_image"
accelerate launch $CONFIG_FILE_ARG \
    -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained="${PRETRAINED_MODEL_PATH},max_pixels=147456,attn_implementation=flash_attention_2,interleave_visuals=False" \
    --tasks $TASKS_STR \
    --verbosity WARNING \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $RUN_NAME \
    --output_path $LOG_DIR/$RUN_NAME

# TASKS=(mme mmbench \
# mathvision_test mathvista \
# seedbench_2_plus ocrbench \
# mia_bench \
# scienceqa \
# pope hallusion_bench_image)
# for i in "${!TASKS[@]}"; do
#     TASK=${TASKS[$i]}
#     echo "------------Task $TASK is running------------"
#     accelerate launch \
#         -m lmms_eval \
#         --model qwen2_5_vl \
#         --model_args pretrained="${PRETRAINED_MODEL_PATH},max_pixels=147456,interleave_visuals=False" \
#         --tasks $TASK \
#         --verbosity WARNING \
#         --batch_size 1 \
#         --log_samples \
#         --log_samples_suffix $RUN_NAME \
#         --output_path $LOG_DIR/$RUN_NAME
#     sleep 5
# done
