#!/bin/bash

unset http_proxy https_proxy

#running
model=../zh-pt830-sft
model=../llama-2-zh/chinese-alpaca-2-13b

export CUDA_VISIBLE_DEVICES=7

name='text-davinci-003'
gpu_num=1

echo "runing ${model} with name ${name}"
#nohup python fastchat/serve/model_worker.py --model-path ${model} --model-name ${name} --limit-worker-concurrency 30 --controller-address http://127.0.0.1:21001 --num-gpus ${gpu_num} --max-gpu-memory 35GiB --host 127.0.0.1 --port 31000 --worker http://127.0.0.1:31000  >  ./logs/model_worker.log 2>&1 &

echo "running with vllm worker to accelerate"
nohup python fastchat/serve/vllm_worker.py --model-path ${model} --model-names ${name} --limit-worker-concurrency 300 --controller-address http://127.0.0.1:21001 --num-gpus ${gpu_num} --conv-template vicuna_v1.1 --host 127.0.0.1 --port 31000 --worker-address http://127.0.0.1:31000 --gpu-memory-utilization 0.8 >  ./logs/vllm_worker.log 2>&1 &
