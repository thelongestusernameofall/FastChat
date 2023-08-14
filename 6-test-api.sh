#!/bin/bash

#host=10.178.11.72
#port=8001

host=10.163.166.72
port=81
model='text-davinci-002'
curl http://${host}:${port}/v1/chat/completions \
          -H "Content-Type: application/json" \
          -d "{
          \"model\": \"${model}\",
          \"messages\": [{\"role\": \"user\", \"content\": \"Hello! What is your name?\"}]
}"
