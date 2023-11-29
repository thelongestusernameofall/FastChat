#!/bin/bash
# train pretrain and merge
#
# lora pretraining has poorly effectivness. Infact, it's deprecated and not allowed in the project
#

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#base_model=../llama-2-zh/chinese-alpaca-2-13b

export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=ERROR

base_model=../zh-pt829-pt
lora_name=../zh-pt830
target_name=../zh-pt830
data_path=../data-using
epochs=2
batch_size=2
max_length=512
lora=False
clear_cache=False

#lora_target_modules="q_proj, v_proj, k_proj, o_proj, gate_proj, down_proj, up_proj"
lora_target_modules='q_proj, v_proj, k_proj, o_proj'

layers='all'
layers="layers.3 layers.4"

if [[ "$lora" == "True" ]]; then
    echo "Using lora"
else
    echo "Not using lora"
    lora_name=${target_name}
fi

# Check for the --overwrite flag
# 如果参数中包含 --overwrite
if [[ "$@" =~ "--overwrite" ]]; then
    if [ -d "${lora_name}" ] || [ -f "${lora_name}" ]; then
        rm -rf "${lora_name}"
    fi
    if [ -d "${target_name}" ] || [ -f "${target_name}" ]; then
        rm -rf "${target_name}"
    fi
# 如果参数中包含 --resume
elif [[ "$@" =~ "--resume" ]]; then
    if [ -d "${lora_name}" ] || [ -f "${lora_name}" ]; then
        echo "目录或文件 ${lora_name} 已经存在, 将检测checkpoint并恢复训练"
    fi
    if [ -d "${target_name}" ] || [ -f "${target_name}" ]; then
        echo "目录或文件 ${target_name} 已经存在， 将检测checkpoint并恢复训练"
    fi
else
    if [ -d "${lora_name}" ] || [ -f "${lora_name}" ]; then
        echo "目录或文件 ${lora_name} 已经存在，请删除或使用--overwrite/--resume参数"
        exit 1
    fi
    if [ -d "${target_name}" ] || [ -f "${target_name}" ]; then
        echo "目录或文件 ${target_name} 已经存在，请删除或使用--overwrite/--resume参数"
        exit 1
    fi
fi


deepspeed --hostfile=/mnt/data/run/hostfile fastchat/train/train_pt_layers.py \
    --model_name_or_path ${base_model}  \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules ${lora_target_modules} \
    --data_path ${data_path} \
    --file_type "json" \
    --data_cache_dir ../temp_data_cache_dir \
    --block_size ${max_length} \
    --worker_num 200 \
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
    --flash_attn False \
    --clear_cache ${clear_cache} \
    --layers ${layers}


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


