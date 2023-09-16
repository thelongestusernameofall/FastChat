#!/bin/bash
# train lora and merge

#base_model=../llama-2-zh/chinese-alpaca-2-13b-sft831
base_model=../codellama/CodeLlama-13b-Instruct-hf
lora_name=../codellama/CodeLlama-13b-Instruct-hf-sft915-lora
sft_name=../codellama/CodeLlama-13b-Instruct-hf-sft915
data_path=../data-sft/name.json
epochs=10
batch_size=24
#conv_name="vicuna_v1.1"
conv_name="llama-2"
max_length=1024

#lora_target_modules="q_proj, v_proj, k_proj, o_proj, gate_proj, down_proj, up_proj"
lora_target_modules='q_proj, v_proj, k_proj, o_proj'

deepspeedconf=deepspeed.json
deepspeedconf=playground/deepspeed_config_s2.json

lr=2e-5
lr=2e-4

unset http_proxy && unset https_proxy
# Check for the --overwrite flag
if [[ "$1" == "--overwrite" ]]; then
    if [ -d "${lora_name}" ] || [ -f "${lora_name}" ]; then
        rm -rf "${lora_name}"
    fi
    if [ -d "${sft_name}" ] || [ -f "${sft_name}" ]; then
        rm -rf "${sft_name}"
    fi
else
    if [ -d "${lora_name}" ] || [ -f "${lora_name}" ]; then
        echo "目录或文件 ${lora_name} 已经存在"
        exit 1
    fi
    if [ -d "${sft_name}" ] || [ -f "${sft_name}" ]; then
        echo "目录或文件 ${sft_name} 已经存在"
        exit 1
    fi
fi

deepspeed fastchat/train/train_lora.py \
    --model_name_or_path ${base_model}  \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules ${lora_target_modules} \
    --data_path ${data_path} \
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
    --learning_rate ${lr} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length ${max_length} \
    --q_lora False \
    --deepspeed ${deepspeedconf} \
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


