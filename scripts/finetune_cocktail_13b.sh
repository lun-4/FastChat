#!/bin/sh
#
# NOTE: make sure to set gradient_checkpointing to False if you have
# enough RAM to accomodate it. setting it to True requires some code
# changes that i am not ready to do

set -eux

model=$1
dataset_path=$2
path_to_deepspeed=${3:-deepspeed}

$path_to_deepspeed fastchat/train/train_lora.py \
    --deepspeed ./ds_config.json \
    --model_name_or_path $model \
    --data_path $dataset_path \
    --bf16 True \
    --output_dir output_vicuna_13b \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_steps 1500 \
    --save_strategy "steps" \
    --save_steps 1500 \
    --save_total_limit 8 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap offload" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True

