#!/bin/bash

#host=10.178.11.72
#port=8001

unset http_proxy https_proxy

host="127.0.0.1"
port=81
model='text-davinci-003'
curl http://${host}:${port}/v1/chat/completions \
          -H "Content-Type: application/json" \
          -d "{
          \"model\": \"${model}\",
          \"messages\": [{\"role\": \"user\", \"content\": \"Hello! What is your name?\"}]
}"


curl http://127.0.0.1:81/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "text-davinci-003",
    "prompt": "USER: Say this is a test. ASSISTANT:",
    "max_tokens": 4096,
    "temperature": 0.7
  }'