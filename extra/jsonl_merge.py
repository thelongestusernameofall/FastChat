import os
import json
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

lock = threading.Lock()  # 全局锁


def process_file(filepath, output_file):
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            content = data.get('Content', '')

            with lock:  # 获取锁
                output_file.write(content + '\n')


def extract_and_save_content(directory, output_filename):
    jsonl_files = [os.path.join(directory, filename) for filename in os.listdir(directory) if
                   filename.endswith('.jsonl') or filename.endswith('.txt')]

    with tqdm(total=len(jsonl_files), desc="Processing files") as pbar:
        with open(output_filename, 'w', encoding='utf-8') as output_file:
            with ThreadPoolExecutor() as executor:
                for _ in executor.map(lambda filepath: process_file(filepath, output_file), jsonl_files):
                    pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge contents of .jsonl files into a single text file.')
    parser.add_argument('-d', '--data_dir', type=str, help='Directory containing .jsonl files', required=True)
    parser.add_argument('-o', '--output', type=str, help='Name of the output file', required=True)

    args = None
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(1)

    extract_and_save_content(args.data_dir, args.output)
