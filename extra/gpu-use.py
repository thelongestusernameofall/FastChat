import torch
import signal
import random
import os, sys
import argparse
import time


def set_gpu_memory_usage(mem_size, sleep=0):
    """Set the GPU memory usage to the specified size (in GB)"""
    torch.cuda.empty_cache()
    x = torch.zeros((int(mem_size * 250 * 0.001), 1024, 1024), dtype=torch.float32, device=torch.device("cuda:0"))
    time.sleep(sleep)
    del x


def signal_handler(sig, frame):
    """Handle the 'ctrl+c' signal and exit gracefully"""
    print("\nExiting...")
    exit(0)


signal.signal(signal.SIGINT, signal_handler)

parser = argparse.ArgumentParser()

# 添加命令行选项
parser.add_argument("--low", help="run low job", action="store_true")
parser.add_argument("--high", help="run high job", action="store_true")

rgl = 100
rgh = 500

args = parser.parse_args()
if args.low:
    rgl = 100
    rgh = 500
elif args.high:
    rgl = 1000
    rgh = 5000

while True:
    # Set the GPU memory usage to a random value between 100MB to 500MB
    mem_size = random.randint(rgl, rgh)
    set_gpu_memory_usage(mem_size, sleep=5)
