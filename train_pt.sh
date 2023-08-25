#!/bin/bash
# train pretrain and merge

base_model=../llama-2-zh/chinese-alpaca-2-13b
lora_name=../zh-pt04-lora
target_name=../zh-pt04-pt
data_path=../pretrain-data
epochs=6
batch_size=1
max_length=512
lora=False
flash_attn=False
#lora_target_modules="q_proj, v_proj, k_proj, o_proj, gate_proj, down_proj, up_proj"
lora_target_modules='q_proj, v_proj, k_proj, o_proj'

if [[ "$lora" == "True" ]]; then
    echo "Using lora"
else
    echo "Not using lora"
    lora_name=${target_name}
fi

# Check for the --overwrite flag
if [[ "$1" == "--overwrite" ]]; then
    if [ -d "${lora_name}" ] || [ -f "${lora_name}" ]; then
        rm -rf "${lora_name}"
    fi
    if [ -d "${target_name}" ] || [ -f "${target_name}" ]; then
        rm -rf "${target_name}"
    fi
else
    if [ -d "${lora_name}" ] || [ -f "${lora_name}" ]; then
        echo "目录或文件 ${lora_name} 已经存在"
        exit 1
    fi
    if [ -d "${target_name}" ] || [ -f "${target_name}" ]; then
        echo "目录或文件 ${target_name} 已经存在"
        exit 1
    fi
fi

deepspeed fastchat/train/train_pt.py \
    --model_name_or_path ${base_model}  \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules ${lora_target_modules} \
    --data_path ${data_path} \
    --data_cache_dir ../temp_data_cache_dir \
    --block_size ${max_length} \
    --worker_num 80 \
    --output_dir ${lora_name} \
    --num_train_epochs ${epochs} \
    --fp16 True \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --eval_steps 100  \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length ${max_length} \
    --q_lora False \
    --lora ${lora} \
    --deepspeed deepspeed.json \
    --gradient_checkpointing True \
    --flash_attn ${flash_attn}


if [[ "$lora" == "True" ]]; then
    echo "Using lora, merging lora"
    # merge lora
    python3 -m fastchat.model.apply_lora --base ${base_model} --target ${target_name} --lora ${lora_name}
else
    echo "Not using lora, copy tokenizer to ${target_name}"
    cp -r ${base_model}/*token* ${target_name}/
fi

# print summary
echo "data path: ${data_path}"
echo "epochs: ${epochs}"
echo "base model: ${base_model}"
if [[ "$lora" == "True" ]]; then
    echo "lora model: ${lora_name}"
fi
echo "target model: ${target_name}"


