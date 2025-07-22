import requests
import time
import json
from termcolor import colored


# 定义一个函数pretty_print_conversation, 用于打印消息对话内容
def pretty_print_conversation(messages):
    # 为不同角色设置不同的颜色
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }

    # 遍历消息列表
    for message in messages:
        # 如果消息的角色是"system", 则用红色打印"content"
        if message["role"] == "system":
            print(colored(f"system: {message['content']}\n", role_to_color[message["role"]]))

        # 如果消息的角色是"user", 则用绿色打印"content"
        elif message["role"] == "user":
            print(colored(f"user: {message['content']}\n", role_to_color[message["role"]]))

        # 如果消息的角色是"assistant", 并且消息中包含"function_call", 则用蓝色打印"function_call"
        elif message["role"] == "assistant" and message.get("function_call"):
            print(colored(f"assistant[function_call]: {message['function_call']}\n", role_to_color[message["role"]]))

        # 如果消息的角色是"assistant", 但是消息中不包含"function_call", 则用蓝色打印"content"
        elif message["role"] == "assistant" and not message.get("function_call"):
            print(colored(f"assistant[content]: {message['content']}\n", role_to_color[message["role"]]))

        # 如果消息的角色是"function", 则用品红色打印"function"
        elif message["role"] == "function":
            print(colored(f"function ({message['name']}): {message['content']}\n", role_to_color[message["role"]]))



url = "http://0.0.0.0:8103/"
data = {"prompt": "你好, 按曲目数量计算, 前五名的艺术家是谁?"}

start_time = time.time()

data = json.dumps(data)

# 向服务发送请求
res = requests.post(url ,data)

cost_time = time.time() - start_time

print('单次查询的耗时:', cost_time, 's')

res = json.loads(res.text)

pretty_print_conversation(res['response'])
