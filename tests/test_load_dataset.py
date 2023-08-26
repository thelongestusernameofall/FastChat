from datasets import load_dataset, concatenate_datasets
import json
import os


def test_load_dataset():
    dataset = load_dataset('json', data_files={'train': 'data/pretrain/little-test.jsonl'})
    # dataset = load_dataset('text', data_files={'train': 'data/pretrain/localbrain-v2.txt'})
    return dataset


if __name__ == '__main__':
    data = test_load_dataset()
    print(data)
