#!/bin/bash
export OPENAI_API_BASE="https://api.360.cn/"
export OPENAI_API_KEY="fk1007450431.l5UjXztYHaGsJ8LQAMjow8L8TbkMtrU469e66468"

host="0.0.0.0"
port=88

# 获取进程号
pid=$(pgrep -f "python3 -m fastchat.serve.gradio_web_server")

# 如果进程存在，则杀死它
if [[ ! -z "$pid" ]]; then
    kill -9 "$pid"
fi

unset http_proxy https_proxy
# 3.2 启动gradio_web_server并添加--add-chatgpt参数
# 下面添加用户登录，测试失败
#  --gradio-auth-path user-auth.txt
#nohup python3 -m fastchat.serve.gradio_web_server --concurrency-count 300 --host ${host} --port ${port} --controller-url http://127.0.0.1:21001 --add-chatgpt --model-list-mode reload --gradio-auth-path user-auth.txt > ./logs/gradio_web_server.log 2>&1 &
nohup python3 -m fastchat.serve.gradio_web_server --concurrency-count 30 --host ${host} --port ${port} --controller-url http://127.0.0.1:21001 --add-chatgpt --model-list-mode reload --gradio-auth-path user-auth.txt > ./logs/gradio_web_server.log 2>&1 &