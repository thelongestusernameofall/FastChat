#!/bin/bash
#
# superadd model specific layers
#

model_name=../llama-2-zh/chinese-alpaca-2-13b-16k
output_dir=../test-16k
layers_add=7

python superadd/superadd.py -m ${model_name} -o ${output_dir} -a ${layers_add}
