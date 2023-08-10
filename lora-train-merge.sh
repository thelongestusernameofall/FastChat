#!/bin/bash
# train lora and merge

base_model=../vicuna-13b-v1.3-sft807-v2
lora_name=../vicuna-13b-v1.3-sft807-v3-lora
sft_name=../vicuna-13b-v1.3-sft807-v3
data_path=../merged-retrain-807-2.json
epochs=6
batch_size=1
#conv_name="vicuna"
conv_name="llama-2"

#deepspeed fastchat/train/train_lora.py \
#    --deepspeed deepspeed.json \
#    --lora_r 8 \
#    --lora_alpha 32 \
#    --lora_dropout 0.05 \
#    --model_name_or_path ${base_model} \
#    --data_path ${data_path} \
#    --bf16 True \
#    --output_dir ${lora_name} \
#    --num_train_epochs ${epochs} \
#    --per_device_train_batch_size ${batch_size} \
#    --per_device_eval_batch_size ${batch_size} \
#    --gradient_accumulation_steps 1 \
#    --evaluation_strategy "no" \
#    --save_strategy "steps" \
#    --save_steps 3000 \
#    --save_total_limit 3 \
#    --learning_rate 2e-5 \
#    --weight_decay 0. \
#    --warmup_ratio 0.03 \
#    --lr_scheduler_type "cosine" \
#    --logging_steps 1 \
#    --tf32 True \
#    --model_max_length 512

deepspeed fastchat/train/train_lora.py \
    --model_name_or_path ${base_model}  \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_path ${data_path} \
    --output_dir ${lora_name} \
    --num_train_epochs ${epochs} \
    --fp16 True \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 100  \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --q_lora False \
    --deepspeed deepspeed.json \
    --gradient_checkpointing True \
    --flash_attn False \
    --conv_name ${conv_name}


# merge lora
python3 -m fastchat.model.apply_lora --base ${base_model} --target ${sft_name} --lora ${lora_name}

# print summary
echo "data path: ${data_path}"
echo "epochs: ${epochs}"
echo "base model: ${base_model}"
echo "lora model: ${lora_name}"
echo "sft model: ${sft_name}"


