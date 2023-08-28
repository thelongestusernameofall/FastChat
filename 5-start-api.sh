#!/bin/bash
unset http_proxy https_proxy

#host=10.178.11.72
host=10.163.166.72
port=81

nohup python3 -m fastchat.serve.openai_api_server --host ${host} --port ${port} > ./logs/api.log 2>&1 &
