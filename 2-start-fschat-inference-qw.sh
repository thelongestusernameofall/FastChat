#!/bin/bash

export RAY_BACKEND_LOG_LEVEL=critical
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 定义变量
# previous version
#model_path=../llama-2-zh/chinese-alpaca-2-13b-act1206-v4-t2
#conv_template="vicuna_v1.1"

# current version
model_path=../QWen/Qwen-72B-Chat-Int4
model_name=inf-davinci-004
conv_template="qwen-7b-chat"
max_model_len=29000

# test version
model_path=../Hongshu/Hongshu-14B-chat-240313-v1
model_name=inf-davinci-004
conv_template="qwen-7b-chat"
max_model_len=29000

#gpu_mem_utilization=0.72
gpu_mem_utilization=0.72

# controller settings
controller_host="127.0.0.1"
controller_port=21001

# api server host and port
api_host='0.0.0.0'
api_port=81

# worker setting
worker_host="127.0.0.1"
worker_port=31001


controller_log="./logs/controller.log"
api_log="./logs/api.log"
worker_log="./logs/inference.log"

# 获取api_log的目录路径
api_log_dir=$(dirname $api_log)
mkdir -p $api_log_dir

gpu_count=$(nvidia-smi -L | wc -l)
echo "Number of GPUs: $gpu_count"

# 通过环境变量CUDA_VISIBLE_DEVICES获取GPU数量
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES is not set. Using all GPUs."
else
    echo "CUDA_VISIBLE_DEVICES is set to $CUDA_VISIBLE_DEVICES."
    gpu_count=$(echo $CUDA_VISIBLE_DEVICES | awk -F, '{print NF}')
    echo "Number of GPUs: $gpu_count"
fi

gpu_num=$gpu_count

# 检查 fastchat.serve.controller 是否运行
if ! pgrep -f "fastchat.serve.controller" > /dev/null; then
    echo "[+] Starting fastchat.serve.controller..."
    nohup python -m fastchat.serve.controller --host $controller_host --port $controller_port > $controller_log 2>&1 &
else
    echo "[.] fastchat.serve.controller is already running."
fi

# 启动 worker
echo "Starting worker..."

# 使用 pgrep 查找满足条件的进程
pids=$(pgrep -f "fastchat.serve.vllm_worker.*${worker_port}")

# 检查是否找到了进程
if [ -n "$pids" ]; then
    echo "Found processes to kill: $pids"
    # 遍历找到的每个进程 ID 并杀死它们
    for pid in $pids; do
        echo "Killing process $pid"
        kill $pid
    done
    sleep 5
else
    echo "No matching processes found."
fi


python -m fastchat.serve.vllm_worker --model-path ${model_path} --model-names ${model_name} --limit-worker-concurrency 1024 --controller-address http://${controller_host}:${controller_port} --num-gpus ${gpu_num} --conv-template ${conv_template} --host ${worker_host} --port ${worker_port} --worker-address http://${worker_host}:${worker_port} --gpu-memory-utilization ${gpu_mem_utilization} --trust-remote-code --enforce-eager --max-model-len ${max_model_len} > ${worker_log} 2>&1 &

# 启动api server
if ! pgrep -f "fastchat.serve.openai_api_server" > /dev/null; then
    echo "Starting api server ..."
    nohup python -m fastchat.serve.openai_api_server --host $api_host --port $api_port --controller-address http://${controller_host}:${controller_port} > ${api_log} 2>&1 &
else
    echo "[.] fastchat.serve.openai_api_server is already running."
fi

echo "Done."
