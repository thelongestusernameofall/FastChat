import os
import json
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import re

lock = threading.Lock()  # 全局锁

"""
乱码检测函数
"""


# 若一行中的非英文字符比例大于阈值，则认为是乱码
def is_garbled_text(line, threshold=0.5):
    total_chars = len(line)

    if total_chars == 0:
        return False
    # 汉字英语数字标点符号
    pattern = re.compile(r'[\x00-\x7F\u4e00-\u9FFF\s。，“”！？；：]')
    common_chars = len(pattern.findall(line))
    uncommon_ratio = 1 - (common_chars / total_chars)
    return uncommon_ratio > threshold


def is_chinese_text(line, threshold=0.5):
    total_chars = len(line)

    if total_chars == 0:
        return False
    # 汉字
    pattern = re.compile(r'[\u4e00-\u9FFF]')
    common_chars = len(pattern.findall(line))  # 汉字个数
    chinese_ratio = float(common_chars / total_chars)
    return chinese_ratio > threshold


def process_file(filepath, output_file, min_len, max_len, only_zh=False):
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            content = data.get('Content', '')
            if is_garbled_text(content):
                continue

            if only_zh and not is_chinese_text(content):
                continue

            len = data.get('Length', 0)
            if len < min_len or len > max_len:
                continue

            with lock:  # 获取锁
                # save data to file
                output_file.write(json.dumps(data, ensure_ascii=False) + '\n')


def extract_and_save_content(directory, output_filename, min_len, max_len, suffix, threads, only_zh):
    suffix_list = ['.jsonl', '.txt', '.json']
    if suffix:
        suffix_list = suffix.split(',')
    suffix_list = [s.strip() for s in suffix_list]

    def is_valid_suffix(filename):
        for asuf in suffix_list:
            if filename.lower().endswith(asuf):
                return True
        return False

    # 使用os.walk()遍历目录及其所有子目录
    jsonl_files = [os.path.join(root, filename)
                   for root, dirs, files in os.walk(directory)
                   for filename in files
                   if is_valid_suffix(filename)]

    max_workers = threads if threads > 0 else None
    with tqdm(total=len(jsonl_files), desc="Processing files") as pbar:
        with open(output_filename, 'w', encoding='utf-8') as output_file:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for _ in executor.map(lambda filepath: process_file(filepath, output_file, min_len, max_len, only_zh),
                                      jsonl_files):
                    pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge contents of .jsonl files into a single jsonl file.')
    parser.add_argument('-i', '--input', type=str, help='Input Directory containing .jsonl files', required=True)
    parser.add_argument('-o', '--output', type=str, help='Name of the output file', required=True)
    parser.add_argument('-t', '--threads', type=int, default=16, help='Number of the threads', required=False)
    parser.add_argument('-s', '--suffix', type=str, help='file suffix to scan in data_dir, split by ,',
                        default='json,txt', required=False)
    parser.add_argument('--min-len', type=int, default=256, help='Min length of content to keep', required=False)
    parser.add_argument('--max-len', type=int, default=1024, help='Max length of content to keep', required=False)
    parser.add_argument('--only-zh', action='store_true', help='Only process Chinese content', required=False)

    args = None
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(1)

    extract_and_save_content(args.input, args.output, args.min_len, args.max_len, args.suffix, args.threads,
                             args.only_zh)
