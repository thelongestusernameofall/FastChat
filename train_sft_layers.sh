#!/bin/bash
#
# sft model specific layers
#

base_model=../test
output_dir=../test-v2
data_path=../data-sft/all-sharegpt-v3-no-imsorry.json

epochs=3
batch_size=8
max_length=2048
lr=2e-5

layers='all'
layers="layers.3,layers.4"

# Check for --overwrite argument
overwrite=false
for arg in "$@"
do
    if [ "$arg" = "--overwrite" ]; then
        overwrite=true
        break
    fi
done

# If output_dir exists and overwrite is false, exit the script
if [ -d "$output_dir" ] && [ "$overwrite" = false ]; then
    echo "Error: Output directory $output_dir already exists. Use --overwrite to overwrite."
    exit 1
fi

# Choose the framework: deepspeed or torchrun
framework=deepspeed # or torchrun

# Switch on the framework
if [ "$framework" = "torchrun" ]; then
    # Run training with torchrun
    torchrun --nproc_per_node=8 --master_port=20001 fastchat/train/train.py \
        --model_name_or_path ${base_model} \
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
        --lazy_preprocess True \
        --layers ${layers}
elif [ "$framework" = "deepspeed" ]; then
    # Run training with deepspeed
    deepspeed fastchat/train/train.py \
        --deepspeed deepspeed.json \
        --model_name_or_path ${base_model} \
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
        --model_max_length ${max_length} \
        --gradient_checkpointing True \
        --lazy_preprocess True \
        --layers ${layers}
else
    echo "Error: Unknown framework specified"
    exit 1
fi

echo "Done, generated model is in ${output_dir}"
