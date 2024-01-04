#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# previous version
#model_path=../llama-2-zh/chinese-alpaca-2-13b-act1206-v4-t2
# current version
model_path=../llama-2-zh/chinese-alpaca-2-13b-act1206-v5-t2
# testing
#model_path=../QWen/Qwen-7B-Chat-act1227-v1f
model_name=text-davinci-003

all_gpu_num=8
host='0.0.0.0'
port=82
gpu_mem_use=0.2

## ray master
#ray start --head
## ray slave
# ray start --address=<ray-head-address>
python -m vllm.entrypoints.openai.api_server --model  $model_path --served-model-name $model_name --tensor-parallel-size $all_gpu_num --host $host --port $port --gpu-memory-utilization ${gpu_mem_use} --trust-remote-code --enforce-eager
