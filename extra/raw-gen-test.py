import requests
import argparse


def send_request(query):
    # 请求的URL
    url = "http://10.178.11.72:7860/run/predict"

    # 请求头部信息
    headers = {
        "Accept": "*/*",
        "Content-Type": "application/json",
        "Origin": "http://10.178.11.72:7860",
        "Cookie": "_ga=GA1.1.1134920905.1690188205; _ga_R1FN4KJKJH=GS1.1.1693479973.49.0.1693479973.0.0.0; _gid=GA1.1.534594865.1693479974",
        "Accept-Language": "en-US,en;q=0.9",
        "Host": "10.178.11.72:7860",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.3 Safari/605.1.15",
        "Referer": "http://10.178.11.72:7860/",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive"
    }

    # 请求的数据
    data = {
        "fn_index": 0,
        "data": [f"USER: {query} ASSISTANT:", "", 0.1, 1, 2, 2, 256],
        "event_data": None,
        "session_hash": "7epfnjqgtvq"
    }

    # 发送POST请求
    response = requests.post(url, headers=headers, json=data)

    # 打印响应
    print(response.status_code)  # 响应状态码
    print(response.text)  # 响应内容

    result = "ERROR"
    if response.status_code == 200:
        result = response.json()["data"][0]
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send a request with a query.")
    parser.add_argument("query", type=str, help="The query string to be sent in the request.")
    args = parser.parse_args()
    send_request(args.query)
