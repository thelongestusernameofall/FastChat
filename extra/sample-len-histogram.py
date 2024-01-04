#!/usr/bin/env python
# coding=utf-8

"""
    统计sample的长度分布
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
from tqdm import tqdm


def get_sample_len(sample: dict):
    conversations = sample.get("conversations", None)
    if conversations:
        user_says = conversations[0].get("value", None)
        gpt_says = conversations[1].get("value", None)
        if user_says and gpt_says:
            return len(user_says) + len(gpt_says)
        else:
            return 0
    return 0


def parallel_map(func, iterable, n_jobs=-1, desc="Processing", unit="task"):
    if n_jobs == -1:
        n_jobs = None

    results = []
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(func, item) for item in iterable]

        # 创建进度条
        with tqdm(total=len(futures), desc=desc, unit=unit) as pbar:
            for future in as_completed(futures):
                # 每当一个future完成，更新进度条
                pbar.update(1)
                results.append(future.result())

    return results


def calculate_lengths(json_file, bin_size=10, output_file=None, jobs=8):
    with open(json_file, 'r') as file:
        data = json.load(file)

    lengths = parallel_map(get_sample_len, data, desc="Calculating lengths", n_jobs=jobs)
    lengths = list(filter(lambda x: x >= 0, lengths))

    max_length = max(lengths)
    min_length = min(lengths)

    bins = list(range(0, max_length + bin_size, bin_size))

    plt.hist(lengths, bins=bins, edgecolor='black')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.title('Length Distribution of JSON Elements')
    plt.grid(True)

    x_ticks = list(range(0, max_length + 1, 500))  # 设置每100个单位一个刻度
    plt.xticks(x_ticks, rotation=45)

    # 标出最大值和最小值
    plt.text(max_length, plt.ylim()[1] * 0.9, f'Max: {max_length}', ha='right')
    plt.text(min_length, plt.ylim()[1] * 0.9, f'Min: {min_length}', ha='left')

    if output_file:
        plt.savefig(output_file)
        print(f"Saved histogram to {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Calculate lengths of JSON elements and generate a histogram')
    parser.add_argument('-i', '--input_file', type=str, help='Path to the input JSON file', required=True)
    parser.add_argument('-b', '--bin_size', type=int, default=10, help='Bin size for the histogram')
    parser.add_argument('-o', '--output_file', type=str, help='Path to save the output histogram image')
    parser.add_argument('-j', '--jobs', type=int, default=8, help='Number of jobs to run in parallel')

    args = parser.parse_args()
    calculate_lengths(args.input_file, args.bin_size, args.output_file, args.jobs)


if __name__ == "__main__":
    main()
