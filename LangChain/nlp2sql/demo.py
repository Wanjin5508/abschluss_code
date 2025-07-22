from termcolor import colored
import json
import openai
import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt
import sqlite3
from openai import OpenAI
import os


os.environ['OPENAI_API_KEY'] = ''
openai.api_key = ''

GPT_MODEL = "gpt-4o"
# 直接访问OpenAI的方式, GPT-4o
client = OpenAI(api_key='')


# 定义一个函数chat_completion_request, 主要用于发送聊天补全请求到OpenAI服务器
@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, functions=None, function_call=None, model=GPT_MODEL):
    # 设定请求的header信息
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai.api_key,
    }

    # 设定请求的JSON数据, 包括GPT模型名和要进行补全的消息
    json_data = {"model": model, "messages": messages}

    # 如果传入了functions, 将其加入到json_data中
    if functions is not None:
        json_data.update({"functions": functions})

    # 如果传入了function_call, 将其加入到json_data中
    if function_call is not None:
        json_data.update({"function_call": function_call})

    # 尝试发送POST请求到OpenAI服务器的chat/completions接口
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        # 返回服务器的响应
        return response

    # 如果发送请求或处理响应时出现异常, 打印异常信息并返回
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


conn = sqlite3.connect("../data/chinook.db")
print("Opened database successfully")


# 返回一个包含所有表名的列表
def get_table_names(conn):
    # 创建一个空的表名列表
    table_names = []
    # 执行SQL查询, 获取数据库中所有表的名字
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    # 遍历查询结果, 并将每个表名添加到列表中
    for table in tables.fetchall():
        table_names.append(table[0])
    
    # 返回表名列表
    return table_names


# 返回一个给定表的所有列名的列表
def get_column_names(conn, table_name):
    # 创建一个空的列名列表
    column_names = []
    # 执行SQL查询, 获取表的所有列的信息
    columns = conn.execute(f"PRAGMA table_info('{table_name}');").fetchall()
    # 遍历查询结果, 并将每个列名添加到列表中
    for col in columns:
        column_names.append(col[1])
    
    # 返回列名列表
    return column_names


# 返回一个字典列表, 每个字典包含一个表的名字和列信息
def get_database_info(conn):
    # 创建一个空的字典列表
    table_dicts = []
    # 遍历数据库中的所有表
    for table_name in get_table_names(conn):
        # 获取当前表的所有列名
        columns_names = get_column_names(conn, table_name)
        # 将表名和列名信息作为一个字典添加到列表中
        table_dicts.append({"table_name": table_name, "column_names": columns_names})

    # 返回字典列表
    return table_dicts


# 获取数据库信息, 并存储为字典列表
database_schema_dict = get_database_info(conn)
# print(database_schema_dict)


# 将数据库信息转换为字符串格式, 方便后续使用
database_schema_string = "\n".join(
    [
        f"Table: {table['table_name']}\nColumns: {', '.join(table['column_names'])}"
        for table in database_schema_dict
    ]
)

# print('----------------------------\n\n\n')
# print(database_schema_string)


# 定义一个功能列表, 其中包含一个功能字典, 该字典定义了一个名为"ask_database"的功能, 用于回答用户关于音乐的问题.
functions = [
    {
        "name": "ask_database",
        "description": "使用这个函数回答用户关于音乐的问题. 要求输出的是完全正规化的SQL查询语句.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": f"""
                            利用SQL查询命令提取用户问题的答案.
                            SQL命令使用如下的数据库schema信息:
                            {database_schema_string}
                            查询结果应该是纯文本格式, 不要返回JSON.
                            """,
                }
            },
            "required": ["query"],
        },
    }
]


# 使用query来查询SQLite数据库的函数
def ask_database(conn, query):
    # 执行查询, 并将结果转换为字符串
    try:
        results = str(conn.execute(query).fetchall())
    # 如果查询失败, 捕获异常并返回错误信息
    except Exception as e:
        results = f"query failed with error: {e}"

    return results


# 执行函数调用
def execute_function_call(message):
    # 判断功能调用的名称是否为"ask_database"
    if message["function_call"]["name"] == "ask_database":
        # 如果是, 则获取功能调用的参数, 这里是SQL查询
        query = json.loads(message["function_call"]["arguments"])["query"]
        # 使用ask_database函数执行查询, 并获取结果
        results = ask_database(conn, query)
    else:
        # 如果功能调用的名称不是"ask_database", 则返回错误信息
        results = f"Error: function {message['function_call']['name']} does not exist"
    
    return results


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



# 创建一个空的消息列表
messages = []

# 向消息列表中添加一个系统角色的消息
messages.append({"role": "system", "content": "通过生成标准化的SQL语句查询Chinook音乐数据库, 来回答用户的问题."})

# 向消息列表中添加一个用户角色的消息
messages.append({"role": "user", "content": "你好, 按曲目数量计算, 前五名的艺术家是谁?"})

# 向消息列表中追加一个用户角色的消息
# messages.append({"role": "user", "content": "曲目数量最多的专辑叫什么名字?"})

# 使用chat_completion_request函数获取聊天响应
chat_response = chat_completion_request(messages, functions)
print(chat_response)

# 从聊天响应中获取助手的消息
assistant_message = chat_response.json()["choices"][0]["message"]

# 将助手的消息添加到消息列表中
messages.append(assistant_message)

# 如果助手的消息中有功能调用
if assistant_message.get("function_call"):
    # 使用execute_function_call函数执行功能调用, 并获取结果
    results = execute_function_call(assistant_message)
    # 将功能的结果作为一个功能角色的消息添加到消息列表中
    messages.append({"role": "function", "name": assistant_message["function_call"]["name"], "content": results})

# 使用 pretty_print_conversation 函数打印对话
pretty_print_conversation(messages)

