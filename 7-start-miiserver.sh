#!/bin/bash

unset http_proxy https_proxy ftp_proxy
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 设置参数
MODEL_NAME="../llama-2-zh/chinese-alpaca-2-13b-16k-inf1120-v17"  # 替换为您的模型名
DEPLOYMENT_NAME="text-davinci-003"                # 或者您选择的任何部署名
RESTFUL_API_PORT=28080                            # 您选择的端口
TENSOR_PARALLEL=8                                 # 张量并行度
REPLICA_NUM=1                                     # 模型副本数量
HOST="0.0.0.0"                                    # 主机
PORT=5555                                         # FastAPI 服务的端口
torch_dist_port=29555                             # torch 分布式训练端口

# 运行 Python 脚本
python extra/MiiServer.py --model_name $MODEL_NAME \
               --deployment_name $DEPLOYMENT_NAME \
               --restful_api_port $RESTFUL_API_PORT \
               --torch_dist_port $torch_dist_port \
               --tensor_parallel $TENSOR_PARALLEL \
               --replica_num $REPLICA_NUM \
               --host $HOST \
               --port $PORT