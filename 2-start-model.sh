#!/bin/bash

unset http_proxy https_proxy

#running
model=../test-1.3-ptv1-sft1
model=../llama-2/Llama-2-70b-chat-hf
model=../llama-2-zh/chinese-alpaca-2-13b-16k
model=../llama-2-zh/chinese-alpaca-2-13b-16k-inf1120-v17
model=../QWen/Qwen-72B-Chat


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3

#name='internlm-chat'
#name='text-davinci-003'
#name='llama-2'
#name='text-davinci-002'
name='qwen-7b-chat'
gpu_num=4

convt="vicuna_v1.1"
convt="llama-2"
convt="qwen-7b-chat"

controller=10.176.205.21
#controller=10.178.11.72

echo "runing ${model} with name ${name}"
#nohup python fastchat/serve/model_worker.py --model-path ${model} --model-name ${name} --limit-worker-concurrency 300 --controller-address http://10.176.205.21:21001 --num-gpus ${gpu_num} --max-gpu-memory 60GiB --host 10.178.11.72 --port 31000 --worker http://10.178.11.72:31000  >  ./logs/model_worker.log 2>&1 &

echo "running with vllm worker to accelerate"
nohup python fastchat/serve/vllm_worker.py --model-path ${model} --model-names ${name} --limit-worker-concurrency 1024 --controller-address http://${controller}:21001 --num-gpus ${gpu_num} --conv-template ${convt} --host 10.178.11.72 --port 31000 --worker-address http://10.178.11.72:31000 --gpu-memory-utilization 0.96 --trust-remote-code >  ./logs/vllm_worker.log 2>&1 &
