#!/usr/bin/env python
# coding=utf-8
"""
将tokenizer转换为fast tokenizer
"""
import argparse
import os.path

from transformers import AutoTokenizer


def convert_to_fast(model_name, output_dir: str = None, bak: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if bak:
        original_tokenizer = os.path.join(model_name, "tokenizer.model")
        if os.path.exists(original_tokenizer):
            os.rename(original_tokenizer, os.path.join(model_name, "tokenizer.model.bak"))

    if output_dir:
        tokenizer.save_pretrained(output_dir)
    else:
        tokenizer.save_pretrained(model_name)


def main():
    parser = argparse.ArgumentParser(description='Convert tokenizer to fast tokenizer.')
    parser.add_argument('-m', '--model_name', type=str, required=True, help='Model name')
    parser.add_argument('-o', '--output_dir', type=str, required=False, help='Fast tokenizer output directory')
    parser.add_argument('-b', '--bak', action='store_true', help='Whether to backup the original tokenizer')
    args = parser.parse_args()

    convert_to_fast(args.model_name, args.output_dir, args.bak)


if __name__ == '__main__':
    main()
