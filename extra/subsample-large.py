import argparse
import concurrent.futures
import json
import glob
import random
import threading  # 导入threading模块来使用Lock
from tqdm import tqdm


def process_file(file, sample_rate, output_lock, output_file):
    """处理单个文件，按给定的采样率采样数据，并直接写入输出文件"""
    with open(file, 'r') as f:
        for line in f:
            if random.random() < sample_rate:
                try:
                    data = json.loads(line)
                    with output_lock:
                        json.dump(data, output_file, ensure_ascii=False)
                        output_file.write('\n')
                except json.JSONDecodeError:
                    pass  # 忽略无效的JSON行


def main(sample_rate, output_file_path):
    # 查找当前目录及子目录下所有的.jsonl文件
    files = glob.glob('**/*.jsonl', recursive=True)

    # 打开输出文件
    with open(output_file_path, 'w') as f_out:
        # 创建一个锁对象，用于同步写入输出文件
        output_lock = threading.Lock()

        # 使用多线程进行文件处理
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 使用tqdm创建一个进度条
            futures = [executor.submit(process_file, file, sample_rate, output_lock, f_out) for file in files]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(files), desc="Processing files"):
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process .jsonl files.')
    parser.add_argument('--sample_rate', type=float, required=True, help='The sampling rate (between 0 and 1)')
    parser.add_argument('--output_file', type=str, required=True, help='Output file path')

    args = parser.parse_args()

    main(args.sample_rate, args.output_file)
