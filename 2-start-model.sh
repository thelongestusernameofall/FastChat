#!/bin/bash

unset http_proxy https_proxy
model=../llama-2-zh/chinese-alpaca-2-13b-sft817-v4

#export CUDA_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=0,1

name='text-davinci-003'
gpu_num=2

echo "runing ${model} with name ${name}"
#nohup python fastchat/serve/model_worker.py --model-path ${model} --model-name ${name} --limit-worker-concurrency 30 --controller-address http://127.0.0.1:21001 --num-gpus ${gpu_num} --max-gpu-memory 35GiB --host 127.0.0.1 --port 31000 --worker http://127.0.0.1:31000  >  ./logs/model_worker.log 2>&1 &

nohup python fastchat/serve/vllm_worker.py --model-path ${model} --model-names ${name} --limit-worker-concurrency 300 --controller-address http://127.0.0.1:21001 --num-gpus ${gpu_num} --conv-template vicuna_v1.1 --host 127.0.0.1 --port 31000 --worker-address http://127.0.0.1:31000  >  ./logs/model_worker.log 2>&1 &
