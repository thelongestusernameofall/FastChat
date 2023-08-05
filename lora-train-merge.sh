#!/bin/bash
# train lora and merge

base_model=../vicuna-13b-v1.3-sft731-v5
lora_name=../vicuna-13b-v1.3-sft731-v6-lora
sft_name=../vicuna-13b-v1.3-sft731-v6
data_path=../merged-all-0731.json
epochs=2

deepspeed fastchat/train/train_lora.py \
    --deepspeed deepspeed.json \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --model_name_or_path ${base_model} \
    --data_path ${data_path} \
    --bf16 True \
    --output_dir ${lora_name} \
    --num_train_epochs ${epochs} \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 512


# merge lora
python3 -m fastchat.model.apply_lora --base ${base_model} --target ${sft_name} --lora ${lora_name}

# print summary
echo "base model: ${base_model}"
echo "lora model: ${lora_name}"
echo "sft model: ${sft_name}"
echo "data path: ${data_path}"
echo "epochs: ${epochs}"


