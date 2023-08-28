#!/bin/bash
unset http_proxy https_proxy

python fastchat/serve/test_message.py --model-name text-davinci-003 --controller-address http://127.0.0.1:21001 --worker-address http://127.0.0.1:31000  --message "who are you?"
