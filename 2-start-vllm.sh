#!/bin/bash

<<<<<<< Updated upstream
#export CUDA_VISIBLE_DEVICES=0,1,2,3
model_path=../llama-2-zh/chinese-alpaca-2-13b-sft831
=======
export CUDA_VISIBLE_DEVICES=0

model_path=../codellama/CodeLlama-13b-Instruct-hf-sft915
>>>>>>> Stashed changes
model_name=text-davinci-003
all_gpu_num=1
host='0.0.0.0'
port=81


## ray master
#ray start --head
## ray slave
# ray start --address=<ray-head-address>
<<<<<<< Updated upstream
python -m vllm.entrypoints.openai.api_server --model  $model_path --served-model-name $model_name --tensor-parallel-size $all_gpu_num --host $host --port $port --gpu-memory-utilization 0.85
=======
python -m vllm.entrypoints.openai.api_server --model  $model_path --served-model-name $model_name --tensor-parallel-size $all_gpu_num --host $host --port $port
>>>>>>> Stashed changes
