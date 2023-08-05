#!/bin/bash

host=10.178.11.72
port=8001
curl http://${host}:${port}/v1/chat/completions \
          -H "Content-Type: application/json" \
          -d '{
          "model": "text-davinci-003",
          "messages": [{"role": "user", "content": "Hello! What is your name?"}]
}'
