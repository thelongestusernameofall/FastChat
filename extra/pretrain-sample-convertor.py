#!/usr/bin/env python
# coding=utf-8

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm


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


def item_to_sample(item: dict):
    # id, content --》 'Content', 'File', 'Length'
    if not "id" in item or not "content" in item:
        return None
    length = len(item["content"])

    sample = {
        "Content": item["content"],
        "File": item["id"],
        "Length": length
    }
    return sample


def main():
    # input_file, output_file, jobs_n
    parser = argparse.ArgumentParser(
        description="Convert pretrain samples from InternLM to Icesword format, and subsampling and spliting into mulitipoles files.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input file path.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file path.")
    parser.add_argument("-t", "--type", type=str, required=False, choices=['json', 'jsonl'], default="jsonl",
                        help="Input file type.")
    parser.add_argument("-j", "--jobs", type=int, required=False, default=16, help="Number of jobs to run in parallel.")
    parser.add_argument("-r", "--ratio", type=float, required=False, default=1, help="Ratio of samples to keep.")
    parser.add_argument("-s", "--split", type=int, required=False, default=1, help="Number of files to split into.")

    args = parser.parse_args()
    input_file = args.input
    output_file = args.output
    jobs_n = args.jobs

    data = []

    if args.type == "json":
        with open(input_file, "r", encoding="UTF8") as f:
            data = json.load(f)
    elif args.type == "jsonl":
        with open(input_file, "r", encoding="UTF8") as f:
            for line in f:
                data.append(json.loads(line))

    print(f"len(data) = {len(data)}")

    samples = parallel_map(item_to_sample, data, n_jobs=jobs_n, desc="Convert", unit="sample")
    samples = [sample for sample in samples if sample is not None]

    # random sampling samples with ratio
    random.shuffle(samples)
    samples = samples[:int(len(samples) * args.ratio)]
    print(f"len(samples) = {len(samples)}")

    if args.type == "json":
        if args.split <= 1:
            with open(output_file, "w", encoding="UTF8") as f:
                json.dump(samples, f, ensure_ascii=False, indent=4)
        else:
            for i in range(args.split):
                with open(f"{output_file}.{i}", "w", encoding="UTF8") as f:
                    json.dump(samples[i::args.split], f, ensure_ascii=False, indent=4)
    elif args.type == "jsonl":
        if args.split <= 1:
            with open(output_file, "w", encoding="UTF8") as f:
                for sample in samples:
                    json.dump(sample, f, ensure_ascii=False)
                    f.write("\n")
        else:
            for i in range(args.split):
                with open(f"{output_file}.{i}", "w", encoding="UTF8") as f:
                    for sample in samples[i::args.split]:
                        json.dump(sample, f, ensure_ascii=False)
                        f.write("\n")

    if args.split <= 1:
        print(f"Done. Output file: {output_file}")
    else:
        print(f"Done. Output files: {[f'{output_file}.{i}' for i in range(args.split)]}")


if __name__ == "__main__":
    main()
