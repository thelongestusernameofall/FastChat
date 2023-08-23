lr=2e-4
lora_rank=64
lora_alpha=128
#lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
lora_trainable="q_proj,v_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

pretrained_model=../llama-2/Llama-2-13b-hf/
chinese_tokenizer_path=../llama-2/Llama-2-13b-hf/
dataset_dir=../pretrain-data
data_cache=../temp_data_cache_dir
per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=1
output_dir=../pretrain-output/
epochs=1
block_size=1024

deepspeed_config_file=chinese-llama-train/ds_zero2_no_offload.json

#torchrun --nnodes 1 --nproc_per_node 1 chinese-llama-train/run_clm_pt_with_peft.py \
deepspeed chinese-llama-train/run_clm_pt_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --data_cache_dir ${data_cache} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --seed $RANDOM \
    --fp16 \
    --num_train_epochs ${epochs} \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 2000 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 80 \
    --block_size ${block_size} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 300 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16 \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False

