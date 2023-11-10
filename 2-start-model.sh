#!/bin/bash

unset http_proxy https_proxy

#running
model=../zh-pt830-sft
model=../codellama/CodeLlama-13b-Instruct-hf
model=../llama-2-zh/chinese-alpaca-2-13b
model=../llama-2/Llama-2-70b-chat-hf
model=../codellama/CodeLlama-13b-Instruct-hf
model=../llama-2/Llama-2-13b-chat-hf
model=../Internlm/internlm-chat-20b
model=../llama-2-zh/chinese-alpaca-2-13b-16k
model=../llama-2-zh/chinese-alpaca-2-1.3b
model=../test-v2

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

name='codellama-13b'
#name='internlm-chat'
name='text-davinci-003'
name='llama-2'
gpu_num=8

convt="vicuna_v1.1"
convt="llama-2"

echo "runing ${model} with name ${name}"
#nohup python fastchat/serve/model_worker.py --model-path ${model} --model-name ${name} --limit-worker-concurrency 30 --controller-address http://127.0.0.1:21001 --num-gpus ${gpu_num} --max-gpu-memory 35GiB --host 127.0.0.1 --port 31000 --worker http://127.0.0.1:31000  >  ./logs/model_worker.log 2>&1 &

echo "running with vllm worker to accelerate"
nohup python fastchat/serve/vllm_worker.py --model-path ${model} --model-names ${name} --limit-worker-concurrency 300 --controller-address http://10.178.145.118:21001 --num-gpus ${gpu_num} --conv-template ${convt} --host 10.178.11.72 --port 31000 --worker-address http://10.178.11.72:31000 --gpu-memory-utilization 0.26 --trust-remote-code >  ./logs/vllm_worker.log 2>&1 &
