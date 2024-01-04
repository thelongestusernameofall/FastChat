#!/bin/bash
unset http_proxy https_proxy

#host=10.178.11.72
host="0.0.0.0"
port=5555

nohup python3 -m fastchat.serve.openai_api_server --host ${host} --port ${port} > ./logs/api.log 2>&1 &
