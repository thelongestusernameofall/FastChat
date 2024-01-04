#!/usr/bin/env python
# coding=utf-8
import glob
import multiprocessing
from collections import Counter
import jieba
from tqdm import tqdm
import fire


def process_file(file_path):
    """处理单个文件，统计词频"""
    word_count = Counter()
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            # 使用jieba进行中文分词
            words = list(jieba.cut(line))
            word_count.update(words)
    return word_count


def main(output: str = "output.txt", top: int = 1000, suffix: str = "txt"):
    # 查找当前目录及子目录下所有的.txt文件
    files = glob.glob(f'**/*.{suffix}', recursive=True)

    # 使用多进程进行文件处理
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    results = list(tqdm(pool.imap(process_file, files), total=len(files), desc="Processing files"))

    # 汇总所有结果
    total_count = sum(results, Counter())

    # 获取频率最高的1000个词
    top_1000_words = total_count.most_common(top)

    # 写入到输出文件
    with open(output, 'w', encoding='utf-8') as output_file:
        for word, _ in top_1000_words:
            output_file.write(word + '\n')


if __name__ == "__main__":
    fire.Fire(main)
