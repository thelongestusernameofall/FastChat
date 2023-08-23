#!/bin/bash

#model=../llama-2/LLaMA-2-7B-32K
model=../llama-2-zh/chinese-alpaca-2-13b
model=../llama-2-zh/chinese-alpaca-2-13b-sft817-v3
model=../vicuna-13b-pt01

#export CUDA_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=1

name='text-davinci-003'
gpu_num=1

echo "runing ${model} with name ${name}"
nohup python fastchat/serve/model_worker.py --model-path ${model} --model-name ${name} --limit-worker-concurrency 30 --controller-address http://127.0.0.1:21001 --num-gpus ${gpu_num} --max-gpu-memory 75GiB --host 127.0.0.1 --port 31000 --worker http://127.0.0.1:31000  >  ./logs/model_worker.log 2>&1 &
