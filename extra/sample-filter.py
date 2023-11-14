import json
import argparse
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


def is_valid_sample(sample):
    # 定义预期的角色顺序
    expected_roles = ["human", "gpt"]

    if not 'conversations' in sample:
        return False

    conv = sample['conversations']
    if not isinstance(conv, list):
        return False
    if len(conv) < 2:
        return False
    try:
        # 检查列表的第一项是否是一个字典，并且包含"from"键
        if "from" not in conv[0]:
            # print(f"Invalid sample (first item is not a dict or missing 'from' key): {sample}")
            return False
    except:
        print(f"sample is {sample}")
        return False

    # 检查对话是否以人类开始
    if conv[0]["from"] != expected_roles[0]:
        return False

    # 检查每一条消息的角色是否与预期的角色相符
    for index, message in enumerate(conv):
        if message["from"] != expected_roles[index % 2]:
            return False

    return True


def sft_to_pretrain(sample, file_name: str = "sft-conversation.json"):
    result = {
        "Content": None,
        "File": file_name,
        "Length": -1
    }
    content = ""
    for message in sample["conversations"]:
        content += message["content"] + "\n"
    result["Content"] = content.strip()
    result["Length"] = len(result["Content"])
    return result


def sample_and_shuffle(input_file, sample_count, output_file, len_limit=4090, n_jobs=8):
    if "," in input_file:
        files = input_file.split(",")
    else:
        files = [input_file]
    data = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            d = json.load(f)
            data.extend(d)

    # 过滤不符合条件的样本
    valid_samples = parallel_map(lambda x: x if is_valid_sample(x) and len(str(x)) < len_limit else None, data,
                                 n_jobs=n_jobs, desc="Filtering Invalid Samples")
    valid_samples = [sample for sample in valid_samples if sample is not None]

    # 如果有效的样本数量小于用户指定的数量，打印警告信息
    if len(valid_samples) < sample_count:
        print("Warning: Not enough valid samples. Sampling all valid samples.")

    # 进行随机抽样和乱序
    if sample_count > 0:
        sampled_data = random.sample(valid_samples, min(sample_count, len(valid_samples)))
    else:
        sampled_data = valid_samples
    random.shuffle(sampled_data)

    if args.type == "pretrain":
        sampled_data = parallel_map(sft_to_pretrain, sampled_data, n_jobs=n_jobs, desc="Converting to Pretrain Format")
        sampled_data = list(sampled_data)

    # 保存到输出的JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sample and shuffle valid samples from a JSON file.")
    parser.add_argument('-i', '--input', required=True, help="Input JSON file name.")
    parser.add_argument('-n', '--number', type=int, required=False, default=-1,
                        help="Number of elements to sample from the input JSON.")
    parser.add_argument('-l', "--length", type=int, required=False, default=4090, help="Length limit of whole sample")
    parser.add_argument('-o', '--output', required=True, help="Output JSON file name.")
    parser.add_argument('-j', '--jobs', type=int, required=False, default=8, help="Number of threads to use.")
    parser.add_argument('-t', '--type', type=str, choices=["sft", "pretrain"], required=False, default="sft",
                        help="type of sample to generate")
    args = parser.parse_args()

    sample_and_shuffle(args.input, args.number, args.output, args.length, args.threads)
