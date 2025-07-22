import openai
from openai import OpenAI
import os


os.environ['OPENAI_API_KEY'] = ''
openai.api_key = ''


# 直接访问OpenAI的方式, GPT-4o
client = OpenAI(api_key='')


# models = client.models.list()
# print(models)
# print('--------------------------------------------------')


# model_list = [model.id for model in models.data]
# print(model_list)
# print(len(model_list))
# print('--------------------------------------------------')

# info = client.models.retrieve("gpt-4o-2024-12-09")
# print(info)



'''
info = client.models.retrieve("gpt-4o")
print(info)
info = client.models.retrieve("gpt-3.5-turbo")
print(info)
info = client.models.retrieve("whisper-1")
print(info)
info = client.models.retrieve("o1-mini")
print(info)
'''





response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages = [
            {
                'role': 'system',
                'content': '你是OpenAI最新最强大的聊天助手,你擅长中文, 英文等其他语言对话, 你能对用户的各种问题给出快速准确的回答.'
            },
            {
                'role': 'user',
                'content': '请复述: 这是一个测试程序.'
            },
        ],
    )

print(response)
print('---------------------------------')
print(response.choices[0].message.content)
print('---------------------------------')


response = client.chat.completions.create(
        model='gpt-4',
        messages = [
            {
                'role': 'system',
                'content': '你是OpenAI最新最强大的聊天助手,你擅长中文, 英文等其他语言对话, 你能对用户的各种问题给出快速准确的回答.'
            },
            {
                'role': 'user',
                'content': '请讲一个关于国足的笑话.'
            },
        ],
    )

print(response.choices[0].message.content)

