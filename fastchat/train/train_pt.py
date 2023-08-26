# Usage: deepspeed train_pt.py --deepspeed <$PATH_TO_DEEPSPEED_CONFIG>
#
#   Simon Added, 2023-08-24
#   pretrain script with lora or full-parameter
#
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import copy
import json
from dataclasses import dataclass, field
import logging
import pathlib
import typing
from typing import Dict, List, Optional, Tuple, Union
import os
from itertools import chain

from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, BitsAndBytesConfig, deepspeed
import torch
import datasets
from datasets import load_dataset, concatenate_datasets

from fastchat.train.train import (
    ModelArguments,
)

from fastchat.train.llama_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)


class PretrainDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 dataset_dir,
                 data_cache_dir,
                 worker_num=8,
                 block_size=1024,
                 debug_mode=False,
                 file_type='text'
                 ):
        super(PretrainDataset, self).__init__()

        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data_cache_dir = data_cache_dir
        self.debug_mode = debug_mode
        self.worker_num = worker_num
        self.file_type = file_type  # text or json, jsonl

        if self.block_size is None or self.block_size <= 0:
            self.block_size = self.tokenizer.model_max_length
            # Our input block size will be the max possible for the model
            if self.block_size > 1024:
                print(
                    f"The input block size is greater than 1024. Setting the max block size to {self.block_size}."
                )
                self.block_size = 1024
        else:
            if self.block_size > self.tokenizer.model_max_length:
                print(
                    f"The input block size is greater than {self.tokenizer.model_max_length}. Using self.tokenizer.model_max_length as block size."
                )
                self.block_size = self.tokenizer.model_max_length

        self.data = self._process_data()

    def _tokenize_function(self, examples):
        # output = self.tokenizer(examples["text"])
        self.tokenizer.add_eos_token = True
        samples = examples["text"] if "text" in examples else examples["Content"] if "Content" in examples else None
        if samples is None:
            raise ValueError("No text field in examples")
        output = self.tokenizer(
            samples,
            return_tensors="pt",
            padding="max_length",
            max_length=self.block_size,
            truncation=True,
        )
        return output

    def _group_texts(self, examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }
        # 很明显<s>token和正文的第一个token不需要预测。
        # labels should be deepcopy of input_ids
        result["labels"] = copy.deepcopy(result["input_ids"])
        for i in range(len(result["labels"])):
            result["labels"][i][:2] = [-100] * 2
        return result

    def _extend_group_texts(self, examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}

        # SimonAdd:
        # concatenated_examples['labels'] = concatenated_examples['input_ids'].copy()
        # logger.info(f"concatenated_examples is {concatenated_examples}")

        items = {}
        for key in concatenated_examples.keys():
            items[key] = []
        items['labels'] = []

        for i in range(len(concatenated_examples['input_ids'])):
            label_i = concatenated_examples['input_ids'].copy()
            label_i[:i] = [-100] * i
            for key in concatenated_examples.keys():
                items[key].append(concatenated_examples[key])
            items['labels'].append(label_i)

        # set first label to -100
        # concatenated_examples['labels'][0] = -100
        # print(f"concatenated_examples is {concatenated_examples}")

        # # Split by chunks of max_len.
        result = {}
        for k, v in items.items():
            for i, t in enumerate(v):
                if k not in result:
                    result[k] = []
                result[k].append(t[:self.block_size])

        return result

    def _process_data(self):
        # The processing code you provided
        # This will load your data, tokenize it, and then group it as you described

        # (your main data processing code here)
        lm_datasets = []
        path = pathlib.Path(os.path.abspath(self.dataset_dir))
        ext = [".txt"] if self.file_type == 'text' else [".jsonl", ".json"]
        # files = [file.name for file in path.glob("*.txt")]
        files = [file.name for file in path.glob("*") if file.suffix in ext]
        if len(files) == 0:
            raise ValueError(f"No files found in the dataset directory {os.path.abspath(self.dataset_dir)}")
        if self.debug_mode is True:
            files = files[:1]
        for idx, file in enumerate(files):
            print(f"Processing file {idx + 1}/{len(files)}: {file}")
            data_file = os.path.join(path, file)
            filename = ''.join(file.split('.')[:-1])
            cache_path = os.path.join(self.data_cache_dir, filename)
            os.makedirs(cache_path, exist_ok=True)
            try:
                processed_dataset = datasets.load_from_disk(cache_path, keep_in_memory=False)
                print(f'training datasets-{filename} has been loaded from disk')
            except Exception:
                cache_dir = os.path.join(self.data_cache_dir, filename + "_text")
                os.makedirs(cache_dir, exist_ok=True)
                self.file_type = 'text' if self.file_type == 'text' else 'json'
                raw_dataset = load_dataset(self.file_type, data_files=data_file, cache_dir=cache_dir,
                                           keep_in_memory=False)
                print(f"{file} has been loaded from disk")

                tokenized_dataset = raw_dataset.map(
                    self._tokenize_function,
                    batched=True,
                    num_proc=self.worker_num,
                    remove_columns="text",
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    cache_file_names={k: os.path.join(cache_dir, 'tokenized.arrow') for k in raw_dataset},
                    desc="Running tokenizer on dataset",
                )
                grouped_datasets = tokenized_dataset.map(
                    self._group_texts,
                    batched=True,
                    num_proc=self.worker_num,
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    cache_file_names={k: os.path.join(cache_dir, 'grouped.arrow') for k in tokenized_dataset},
                    desc=f"Grouping texts in chunks of {self.block_size}",
                )
                print(f"grouped_datasets is {grouped_datasets}")
                processed_dataset = grouped_datasets
                processed_dataset.save_to_disk(cache_path)
            if idx == 0:
                lm_datasets = processed_dataset['train']
            else:
                lm_datasets = concatenate_datasets([lm_datasets, processed_dataset['train']])
        return lm_datasets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return data for a specific index
        return self.data[idx]


# Simon Added, 2023-08-01
def adapt_model_to_tokenizer(model, tokenizer):
    model_vocab_size = model.get_input_embeddings().weight.size(0)
    tokenizer_vocab_size = len(tokenizer)
    if model_vocab_size == 0:
        # 尝试直接从模型获取输入嵌入
        try:
            model_vocab_size = model.embeddings.word_embeddings.weight.size(0)
        except AttributeError:
            try:
                model_vocab_size = model.transformer.wte.weight.size(0)
            except AttributeError:
                print("Unable to directly access embeddings (model.transformer.wte.weight) from model.")
                return model, tokenizer
            print("Unable to directly access embeddings (model.embeddings.word_embeddings.weight) from model.")
            return model, tokenizer

    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenizer_vocab_size}")
    if model_vocab_size != tokenizer_vocab_size:
        assert tokenizer_vocab_size > model_vocab_size
        print("Resize model embeddings to fit tokenizer")
        model.resize_token_embeddings(tokenizer_vocab_size)
    return model, tokenizer


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    file_type: str = field(
        default='text', metadata={"help": "File type of the training data. (text/json)"}
    )
    data_cache_dir: str = field(
        default=None, metadata={"help": "Path to the cache dir."}
    )
    worker_num: int = field(
        default=8, metadata={"help": "parallel worker number to process pretrain corpus *.txt ."}
    )
    block_size: int = field(
        default=1024, metadata={"help": "Block size of the input sequence."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False
    conv_name: str = field(
        default="vicuna", metadata={"help": "Conversation template name."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: typing.Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    flash_attn: bool = False


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: typing.List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    lora: bool = True


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def make_pretrain_data_module(
        tokenizer: transformers.PreTrainedTokenizer, data_args, debug_mode=False
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        PretrainDataset
    )
    print("Loading data...")

    # train_json = json.load(open(data_args.data_path, "r"))
    # train_dataset = dataset_cls(train_json, tokenizer=tokenizer, conv_name=data_args.conv_name)
    train_dataset = dataset_cls(
        tokenizer=tokenizer,
        dataset_dir=data_args.data_path,
        data_cache_dir=data_args.data_cache_dir,
        debug_mode=debug_mode,
        block_size=data_args.block_size,
        # worker_num=data_args.worker_num,
        file_type=data_args.file_type
    )

    eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    if training_args.flash_attn:
        replace_llama_attn_with_flash_attn()

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP and ZeRO3 are both currently incompatible with QLoRA."
            )

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # 加载模型之前，先检查训练样本是否存在
    if not os.path.exists(data_args.data_path):
        raise ValueError(f"Training data path {data_args.data_path} does not exist.")

    path = pathlib.Path(os.path.abspath(data_args.data_path))
    ext = [".txt"] if data_args.file_type == 'text' else [".jsonl", ".json"]
    files = [file.name for file in path.glob("*") if file.suffix in ext]
    if len(files) == 0:
        raise ValueError(f"No files found in the dataset directory {os.path.abspath(data_args.data_path)}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.unk_token

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        if lora_args.q_lora
        else None,
        trust_remote_code=True,
    )
    model, tokenizer = adapt_model_to_tokenizer(model, tokenizer)

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    if training_args.local_rank == 0:
        print(f"lora_config is ----- \n{lora_config}")

    if lora_args.q_lora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )
        if not ddp and torch.cuda.device_count() > 1:
            # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
            model.is_parallelizable = True
            model.model_parallel = True

    if lora_args.lora:
        model = get_peft_model(model, lora_config)

    if training_args.flash_attn:
        for name, module in model.named_modules():
            if "norm" in name:
                module = module.to(compute_dtype)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    module = module.to(compute_dtype)
    if lora_args.lora and training_args.deepspeed is not None and training_args.local_rank == 0:
        model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    data_module = make_pretrain_data_module(tokenizer=tokenizer, data_args=data_args)

    if training_args.local_rank == 0:
        print(f"data_module['train_dataset'][0] = {data_module['train_dataset'][0]}")
        print(
            f"text of data_module['train_dataset'][0] = {tokenizer.decode(data_module['train_dataset'][0]['input_ids'])}")

    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    model.config.use_cache = False

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        # use deepspeed engine internal function to gather state dict
        # state_dict_zero3 contains whole parameters of base and lora adapters
        # we will not extract lora parameters since peft save_pretrained will do that
        # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/peft_model.py#L125
        # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/utils/save_and_load.py#L19
        state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
        if training_args.local_rank == 0:
            state_dict = state_dict_zero3
    else:
        # in other mode we use original code from fastchat team, to make sure our change is minimum
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), lora_args.lora_bias
        )

    if training_args.local_rank == 0:
        model.save_pretrained(training_args.output_dir, state_dict=state_dict)


if __name__ == "__main__":
    train()
