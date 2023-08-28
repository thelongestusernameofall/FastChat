#!/bin/bash
host=10.178.11.72
port=8001

unset http_proxy https_proxy

nohup python3 -m fastchat.serve.openai_api_server --host ${host} --port ${port} > ./logs/api.log 2>&1 &
