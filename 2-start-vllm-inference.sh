#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
model_path=../llama-2-zh/chinese-alpaca-2-13b-sft1029-v3
model_name=text-davinci-003
all_gpu_num=4
host='0.0.0.0'
port=81


## ray master
#ray start --head
## ray slave
# ray start --address=<ray-head-address>
python -m vllm.entrypoints.openai.api_server --model  $model_path --served-model-name $model_name --tensor-parallel-size $all_gpu_num --host $host --port $port --gpu-memory-utilization 0.65