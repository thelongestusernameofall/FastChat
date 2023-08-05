#!/bin/bash
deepspeed fastchat/train/train_lora.py \
    --deepspeed deepspeed.json \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --model_name_or_path ../vicuna-13b-v1.3-sft713 \
    --data_path ../merged-retrain-0711+0712.json \
    --bf16 True \
    --output_dir ../vicuna-13b-v1.3-sft713-v2-lora \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 512
