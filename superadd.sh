#!/bin/bash
#
# superadd model specific layers
#

model_name=../llama-2-zh/chinese-alpaca-2-13b-16k
model_name=../llama-2-zh/chinese-alpaca-2-1.3b
output_dir=../test-1.3
layers_add=1

python superadd/superadd.py -m ${model_name} -o ${output_dir} -a ${layers_add}
