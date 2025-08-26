from openai import OpenAI
# tiktoken是OpenAI开发的一个库, 用于从模型生成的文本中计算token数量
import tiktoken


# 定义函数 num_tokens_from_messages, 该函数返回由一组消息所使用的token数.
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    # 尝试获取模型的编码
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # 如果模型没有找到, 使用cl100k_base编码并给出警告
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    # 针对不同的模型设置token数量
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # 每条消息遵循 {role/name}\n{content}\n 格式
        tokens_per_name = -1    # 如果有名字, 角色会被省略
    elif "gpt-3.5-turbo" in model:
        # 对于 gpt-3.5-turbo 模型可能会有更新, 此处返回假设为 gpt-3.5-turbo-0613 的token数量, 并给出警告
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        # 对于 gpt-4 模型可能会有更新, 此处返回假设为 gpt-4-0613 的token数量, 并给出警告
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    elif model in {
        "davinci",
        "curie",
        "babbage",
        "ada"
        }:
        print("Warning: gpt-3 related model is used. Returning num tokens assuming gpt2.")
        encoding = tiktoken.get_encoding("gpt2")
        num_tokens = 0
        # 只计算content部分
        for message in messages:
            for key, value in message.items():
                if key == "content":
                    num_tokens += len(encoding.encode(value))
        return num_tokens
    else:
        # 对于没有实现的模型, 抛出未实现错误
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    # 计算每条消息的token数
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    
    # 每条回复都以助手为首
    num_tokens += 3
    return num_tokens


# 让我们验证上面的函数是否与OpenAI API的响应匹配
example_messages = [
    {
        "role": "system",
        "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English.",
    },
    {
        "role": "system",
        "name": "example_user",
        "content": "New synergies will help drive top-line growth.",
    },
    {
        "role": "system",
        "name": "example_assistant",
        "content": "Things working well together will increase revenue.",
    },
    {
        "role": "system",
        "name": "example_user",
        "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage.",
    },
    {
        "role": "system",
        "name": "example_assistant",
        "content": "Let's talk later when we're less busy about how to do better.",
    },
    {
        "role": "user",
        "content": "This late pivot means we don't have time to boil the ocean for the client deliverable.",
    },
]


# 直接访问OpenAI的方式
# client = OpenAI


'''
for model in [
    # "gpt-3.5-turbo-0613",
    # "gpt-3.5-turbo",
    # "gpt-4-0613",
    "gpt-4",
    ]:
    print(model)
    print(f"{num_tokens_from_messages(example_messages, model)} prompt tokens counted by num_tokens_from_messages().")
    
    # OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=example_messages,
    )
    print(f'{response.usage.prompt_tokens} prompt tokens counted by the OpenAI API.')
    print()
'''


# model = 'gpt-3.5-turbo-0613'
# print(model)
# print(f"{num_tokens_from_messages(example_messages, model)} prompt tokens counted by num_tokens_from_messages().")


'''
# 访问国内最先进的阶跃星辰的Step-2大模型



response = client.chat.completions.create(
        model='step-2-16k',
        messages=example_messages,
        stream=False,
    )



# print('step-1-8k')
print('step-2-16k')
# print('step-1-flash')
print(response)
print(f"{response.usage.prompt_tokens} prompt tokens counted by step-1-8k.")
print(f"{response.usage.completion_tokens} completion tokens counted by step-1-8k.")
print(f"{response.usage.total_tokens} total tokens counted by step-1-8k.")
'''

