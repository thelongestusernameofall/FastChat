from datasets import load_dataset, concatenate_datasets
import json
import os

def test_load_dataset():
    dataset = load_dataset('json', data_files={'train': 'tests/fixtures/sample.json'})
    assert dataset['train'][0]['text'] == 'This is a sample text.'
    assert dataset['train'][0]['label'] == 0