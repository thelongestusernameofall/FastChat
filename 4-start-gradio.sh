#!/bin/bash
#nohup python3 -m fastchat.serve.gradio_web_server --concurrency-count 30 --host 10.163.166.72 --port 80 --controller-url http://127.0.0.1:21001 > ./logs/gradio_web_server.log 2>&1 &

export OPENAI_API_BASE="https://ai.360.cn/api/v1"
export OPENAI_API_KEY="fk1007450431.xPaaYZE9yV93CugliD6CurdDzuaWRP7geed60c0f"

#host="10.178.11.72"
host="10.163.166.72"
port=80

unset http_proxy https_proxy
# 3.2 启动gradio_web_server并添加--add-chatgpt参数
# 下面添加用户登录，测试失败
#  --gradio-auth-path user-auth.txt
nohup python3 -m fastchat.serve.gradio_web_server --concurrency-count 30 --host ${host} --port ${port} --controller-url http://127.0.0.1:21001 --add-chatgpt --model-list-mode reload > ./logs/gradio_web_server.log 2>&1 &
