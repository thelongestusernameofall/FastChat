#!/bin/bash
# train lora and merge

#base_model=../llama-2-zh/chinese-alpaca-2-7b
#lora_name=../llama-2-zh/chinese-alpaca-2-7b-act1223-v1-lora
#sft_name=../llama-2-zh/chinese-alpaca-2-13b-16k-inf1120-v17
#sft_name=../llama-2-zh/chinese-alpaca-2-7b-act1223-v1

#base_model=../llama-2-zh/chinese-alpaca-2-13b-sft1102-v6-t6
#base_model=../llama-2-zh/chinese-alpaca-2-13b-act1206-v2-t2
base_model=../llama-2-zh/chinese-alpaca-2-13b-act1206-v4-t2
lora_name=../llama-2-zh/chinese-alpaca-2-13b-act1206-v5-t2-lora
sft_name=../llama-2-zh/chinese-alpaca-2-13b-act1206-v5-t2

data_path=../data-sft/a+b-1225-sample.json

epochs=2
batch_size=1
conv_name="vicuna_v1.1"
#conv_name="llama-2"
max_length=2048

#lora_target_modules="q_proj, v_proj, k_proj, o_proj, gate_proj, down_proj, up_proj"
#lora_target_modules='q_proj, v_proj, up_proj, down_proj'
lora_target_modules='q_proj, v_proj'
#lora_target_modules='layers.4.self_attn.q_proj, layers.4.self_attn.k_proj, layers.4.self_attn.v_proj, layers.4.self_attn.o_proj, layers.4.mlp.gate_proj, layers.4.mlp.up_proj, layers.4.mlp.down_proj'

deepspeedconf=deepspeed_s3.json
deepspeedconf=deepspeed-cpu.json
deepspeedconf=playground/deepspeed_config_s3.json
deepspeedconf=deepspeed_s3.json

lr=2e-5
#lr=2e-4

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
    --lora_r 16 \
    --lora_alpha 32 \
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
    --save_total_limit 1 \
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
    --lazy_preprocess True \
    --conv_name ${conv_name}


# merge lora
python3 -m fastchat.model.apply_lora --base ${base_model} --target ${sft_name} --lora ${lora_name}

# print summary
echo "data path: ${data_path}"
echo "epochs: ${epochs}"
echo "base model: ${base_model}"
echo "lora model: ${lora_name}"
echo "sft model: ${sft_name}"


