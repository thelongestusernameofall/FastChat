#!/bin/bash
nohup python fastchat/serve/controller.py --host 127.0.0.1 > ./logs/controller.log 2>&1 &