#!/bin/bash
# merge lora list

# lora_list is a list of lora models
base_model=../llama-2-zh/chinese-alpaca-2-13b-sft923-v2
lora_list=(../llama-2-zh/chinese-alpaca-2-13b-sft923-v3-lora ../llama-2-zh/chinese-alpaca-2-13b-sft923-v4-lora ../llama-2-zh/chinese-alpaca-2-13b-sft923-v5-lora)
output_model=../llama-2-zh/chinese-alpaca-2-13b-sft923-v5

temp_sft=../temp-sft
temp_base=../temp-base

# get absolute path of base_model and temp_base
base_model=$(realpath ${base_model})
temp_base=$(realpath ${temp_base})
temp_sft=$(realpath ${temp_sft})
output_model=$(realpath ${output_model})

# delete temp_sft and temp_base if exists
if [ -d ${temp_sft} ]; then
    rm -rf ${temp_sft}
fi

if [ -d ${temp_base} ]; then
    rm -rf ${temp_base}
fi



# soft link base model to temp_base
ln -s ${base_model} ${temp_base}

ls -l ${temp_base}

# loop on lora_list
for lora in ${lora_list[@]}; do
    # get absolute path of lora
    lora=$(realpath ${lora})
    echo "merge lora: ${lora}"

    # check if temp_base exists
    if [ ! -d ${temp_base} ]; then
        echo "temp_base not exists"
        exit 1
    fi

    # merge lora
    python3 -m fastchat.model.apply_lora --base ${temp_base} --target ${temp_sft} --lora ${lora}

    # remove temp_base
    rm -rf ${temp_base}
    # make sure temp_base is removed
    if [ -d ${temp_base} ]; then
        echo "temp_base not removed"
        exit 1
    fi

    # check if temp_sft exists
    if [ ! -d ${temp_sft} ]; then
        echo "temp_sft not exists"
        exit 1
    fi

    # move temp_sft to temp_base
    mv ${temp_sft} ${temp_base}

    # make sure temp_sft is moved
    if [ -d ${temp_sft} ]; then
        echo "temp_sft not moved"
        exit 1
    fi

done

# move temp_base to output_model
mv ${temp_base} ${output_model}

# print summary
echo "base model: ${base_model}"
echo "lora list: ${lora_list[@]}"
echo "output model: ${output_model}"
