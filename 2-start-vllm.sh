#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_path=../codellama/CodeLlama-13b-Instruct-hf
model_name=text-davinci-003
all_gpu_num=1
host='0.0.0.0'
port=81


## ray master
#ray start --head
## ray slave
# ray start --address=<ray-head-address>
python -m vllm.entrypoints.openai.api_server --model  $model_path --served-model-name $model_name --tensor-parallel-size $all_gpu_num --host $host --port $port --gpu-memory-utilization 0.85
