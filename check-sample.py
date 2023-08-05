import json

def check_conversations_key(json_file_path):
    with open(json_file_path, "r") as file:
        data = json.load(file)

    for index, dictionary in enumerate(data):
        if "conversations" not in dictionary:
            print(f"Dictionary at index {index} does not contain 'conversations' key:")
            print(dictionary)

if __name__ == "__main__":
    json_file_path = "../merged-all-0731.json"  # Replace with the actual path to your JSON file
    check_conversations_key(json_file_path)

