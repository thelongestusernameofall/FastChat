#!/bin/bash

#model=../llama-2/LLaMA-2-7B-32K
model=../llama-2/Llama-2-13b-chat-hf

#export CUDA_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=0

#name='text-davinci-003'
name='simon-llama-2-7b'
gpu_num=1

echo "runing ${model} with name ${name}"
nohup python fastchat/serve/model_worker.py --model-path ${model} --model-name ${name} --limit-worker-concurrency 30 --controller-address http://127.0.0.1:21001 --num-gpus ${gpu_num} --max-gpu-memory 75GiB --host 127.0.0.1 --port 31000 --worker http://127.0.0.1:31000  >  ./logs/model_worker.log 2>&1 &
