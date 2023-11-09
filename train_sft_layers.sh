#!/bin/bash
#
# sft model specific layers
#
#deepspeed fastchat/train/train_lora.py \
#    --deepspeed deepspeed.json \
#    --lora_r 8 \
#    --lora_alpha 32 \
#    --lora_dropout 0.05 \
#    --model_name_or_path ../vicuna-13b-v1.3-sft713 \
#    --data_path ../merged-retrain-0711+0712.json \
#    --bf16 True \
#    --output_dir ../vicuna-13b-v1.3-sft713-v2-lora \
#    --num_train_epochs 5 \
#    --per_device_train_batch_size 8 \
#    --per_device_eval_batch_size 8 \
#    --gradient_accumulation_steps 1 \
#    --evaluation_strategy "no" \
#    --save_strategy "steps" \
#    --save_steps 2000 \
#    --save_total_limit 3 \
#    --learning_rate 2e-5 \
#    --weight_decay 0. \
#    --warmup_ratio 0.03 \
#    --lr_scheduler_type "cosine" \
#    --logging_steps 1 \
#    --tf32 True \
#    --model_max_length 512

base_model=../test
output_dir=../test-v2
data_path=../data-sft/all-sharegpt-v3-no-imsorry.json

epochs=3
batch_size=8
max_length=2048
lr=2e-5

torchrun --nproc_per_node=8 --master_port=20001 fastchat/train/train.py \
    --model_name_or_path ${base_model}  \
    --data_path ${data_path} \
    --fp16 True \
    --output_dir ${output_dir} \
    --num_train_epochs ${epochs} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1200 \
    --save_total_limit 10 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length ${max_length} \
    --gradient_checkpointing True \
    --lazy_preprocess True
