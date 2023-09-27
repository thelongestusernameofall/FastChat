#!/bin/bash

unset http_proxy https_proxy

nohup python fastchat/serve/controller.py --host 0.0.0.0 > ./logs/controller.log 2>&1 &
