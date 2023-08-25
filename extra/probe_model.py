#!env python3
# -*- coding: utf-8 -*-

"""
探索模型的内部结构，对模型进行一些测试等
"""

import torch
import argparse


def load_model_and_print_state_dict(model_path):
    model = torch.load(model_path)
    state_dict = model.state_dict()
    for key, value in state_dict.items():
        print(f"{key}: {value.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a PyTorch model and print its state_dict.")
    parser.add_argument("-p", "--path", type=str, required=True, help="Path to the PyTorch model.")
    args = parser.parse_args()

    load_model_and_print_state_dict(args.path)
