#!/usr/bin/env python
# coding=utf-8

#
#   extend model layers
#

import os
import sys
import argparse


def extend_model(model_path: str, output_path: str, model_type: str):
    """
    extend model layers
    :param model_path: model path
    :param output_path: output path
    :param model_type: model type
    :return:
    """
    print("extend model layers")
    # print args for debugging
    print("model_path: ", model_path)
    print("output_path: ", output_path)
    print("model_type: ", model_type)


def main():
    parser = argparse.ArgumentParser(description="supperadd: extend model layers")
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="model name or path containing .pt/.pth files.")
    parser.add_argument("-o", "--output", type=str, required=True, help="output path")
    parser.add_argument("-t", "--type", choices=["hf", "pth"], required=True,
                        help="Specify model type: 'hf' for HuggingFace model, 'pth' for PyTorch .pt or .pth files.")

    args = parser.parse_args()

    # call extend_model
    extend_model(args.model, args.output, args.type)
