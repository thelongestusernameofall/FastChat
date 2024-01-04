#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
model_path=../llama-2-zh/chinese-alpaca-2-13b-inf1029-v6-t6
model_path=../llama-2-zh/chinese-alpaca-2-13b-16k-inf1120-v17
model_path=../QWen/Qwen-72B-Chat

model_name=text-davinci-002
model_name=text-davinci-004

all_gpu_num=8
host='0.0.0.0'
port=81


## ray master
#ray start --head
## ray slave
# ray start --address=<ray-head-address>
python -m vllm.entrypoints.openai.api_server --model  $model_path --served-model-name $model_name --tensor-parallel-size $all_gpu_num --host $host --port $port --gpu-memory-utilization 0.50 --max-model-len 32000 --enforce-eager
