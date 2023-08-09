#!/bin/bash

model=/home/huangshaomang/research/AI/GPT/LLama2/theBloke/models/Llama-2-70B-Chat-fp16
model=/home/huangshaomang/research/AI/GPT/LLama2/linkSoul/Chinese-Llama-2-7b
model=../vicuna-13b-v1.3-sft724
model=../vicuna-13b-v1.3-sft725-v3
model=/home/huangshaomang/research/AI/GPT/LLama2/meta/llama2-hf/Llama-2-7b-chat-hf
model=/home/huangshaomang/research/AI/GPT/LLama2/meta/llama2-hf/Llama-2-70b-hf
model=../llama-2/LLaMA-2-7B-32K
model=../vicuna-13b-v1.3-extoken-sft805
echo "runing ${model}"

#export CUDA_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=0
gpu_num=1
nohup python fastchat/serve/model_worker.py --model-path ${model} --model-name 'text-davinci-003' --limit-worker-concurrency 30 --controller-address http://127.0.0.1:21001 --num-gpus ${gpu_num} --max-gpu-memory 75GiB --host 127.0.0.1 --port 31000 --worker http://127.0.0.1:31000  >  ./logs/model_worker.log 2>&1 &
