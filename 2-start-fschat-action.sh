#!/bin/bash

# 定义变量
# previous version
#model_path=../llama-2-zh/chinese-alpaca-2-13b-act1206-v4-t2
# current version
model_path=../llama-2-zh/chinese-alpaca-2-13b-act1206-v5-t2
# testing
#model_path=../QWen/Qwen-7B-Chat-act1227-v1f
model_name=text-davinci-003
conv_template="vicuna_v1.1"

# api server host and port
host='0.0.0.0'
port=81

worker_port=31000

gpu_mem_utilization=0.2

controller_log="./logs/controller.log"
api_log="./logs/api.log"
worker_log="./logs/action.log"


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
    nohup python -m fastchat.serve.controller --host $host --port 21001 > $controller_log 2>&1 &
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
else
    echo "No matching processes found."
fi


python -m fastchat.serve.vllm_worker --model-path ${model_path} --model-names ${model_name} --limit-worker-concurrency 1024 --controller-address http://127.0.0.1:21001 --num-gpus ${gpu_num} --conv-template ${conv_template} --host ${host} --port ${worker_port} --worker-address http://127.0.0.1:${worker_port} --gpu-memory-utilization ${gpu_mem_utilization} --trust-remote-code > ${worker_log} 2>&1 &

# 启动api server
if ! pgrep -f "fastchat.serve.openai_api_server" > /dev/null; then
    echo "Starting api server ..."
    nohup python -m fastchat.serve.openai_api_server --host $host --port $port --controller-address http://127.0.0.1:21001 > ${api_log} 2>&1 &
else
    echo "[.] fastchat.serve.openai_api_server is already running."
fi

echo "Done."
