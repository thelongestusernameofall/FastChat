#!/usr/bin/env python
# coding=utf-8

#
#   extend model layers
#

import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def extend_model(model_path: str, output_path: str, model_type: str, add_layer: int = 1):
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

    ## load model
    if model_type == "hf":
        model = AutoModelForCausalLM.from_pretrained(model_path)
    elif model_type == "pth":
        model = torch.load(model_path)
    else:
        raise ValueError("Invalid model type: " + model_type)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    config = model.config

    # print model config
    print("model config: ", config)
    print("nubmer of layers: ", config.num_hidden_layers)
    config.num_hidden_layers += add_layer
    print("nubmer of layers after adding: ", config.num_hidden_layers)

    new_model = AutoModelForCausalLM.from_config(config)
    new_model_dict = new_model.state_dict()
    model_dict = model.state_dict()

    # copy weights
    for key in model_dict.keys():
        if key in new_model_dict.keys():
            new_model_dict[key] = model_dict[key]
        else:
            print("key not found in new model: ", key)

    new_model.load_state_dict(new_model_dict)

    # save model acording to model type
    if model_type == "hf":
        new_model.save_pretrained(output_path)
    elif model_type == "pth":
        torch.save(new_model, output_path)
    else:
        raise ValueError("Invalid model type: " + model_type)

    # save tokenizer
    tokenizer.save_pretrained(output_path)

    print("model saved to: ", output_path)


def main():
    parser = argparse.ArgumentParser(description="superadd: extend model layers")
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="model name or path containing .pt/.pth files.")
    parser.add_argument("-o", "--output", type=str, required=True, help="output path")
    parser.add_argument("-a", "--add_layer", type=int, required=True, help="number of layers to add")
    parser.add_argument("-t", "--type", choices=["hf", "pth"], required=False, default="hf",
                        help="Specify model type: 'hf' for HuggingFace model, 'pth' for PyTorch .pt or .pth files.")

    args = parser.parse_args()

    # call extend_model
    extend_model(args.model, args.output, args.type, args.add_layer)


if __name__ == "__main__":
    main()
