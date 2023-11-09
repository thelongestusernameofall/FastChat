#!env python3
# -*- coding: utf-8 -*-

"""
探索模型的内部结构，对模型进行一些测试等
"""
# !env python3
# -*- coding: utf-8 -*-

import os
import torch
import argparse
from transformers import AutoModel


def print_state_dict_of_model(model):
    state_dict = model.state_dict()
    for key, value in state_dict.items():
        print(f"{key}: {value.shape}")


def print_named_parameters_of_model(model):
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")


def load_and_print_hf_model(model_dir):
    model = AutoModel.from_pretrained(model_dir)
    print("=================[model_struct]========================")
    print(f"model: {model}")
    print("=================[state_dict]========================")
    print_state_dict_of_model(model)
    print("=================[named_parameters]========================")
    print_named_parameters_of_model(model)


def load_and_print_pt_model(model_path):
    model = torch.load(model_path)
    print_state_dict_of_model(model)


def load_and_print_pt_models_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.pt') or filename.endswith('.pth'):
            file_path = os.path.join(directory, filename)
            print(f"\nLoading and printing state_dict for {file_path}:")
            load_and_print_pt_model(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a model and print its state_dict.")
    parser.add_argument("-p", "--path", type=str, required=True,
                        help="Path to the model or directory containing .pt/.pth files.")
    parser.add_argument("-t", "--type", choices=["hf", "pth"], required=True,
                        help="Specify model type: 'hf' for HuggingFace model, 'pth' for PyTorch .pt or .pth files.")

    args = parser.parse_args()

    if args.type == "hf":
        load_and_print_hf_model(args.path)
    elif args.type == "pth":
        if os.path.isdir(args.path):
            load_and_print_pt_models_in_directory(args.path)
        else:
            load_and_print_pt_model(args.path)
