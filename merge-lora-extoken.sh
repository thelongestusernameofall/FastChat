#!/bin/bash
#python merge_llama_with_chinese_lora.py --base_model ./chinese-llama-plus --lora_model ./output/pt_lora_model --output_type huggingface --output_dir llama-zh-plus-pt01


base_model=../vicuna-13b-v1.3-extoken
lora_model=../vicuna-13b-v1.3-extoken-sft805-lora
output_dir=../vicuna-13b-v1.3-extoken-sft805
python merge-lora-extoken.py --base_model ${base_model} --lora_model ${lora_model} --output_type huggingface --output_dir ${output_dir}