import os
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory


# 在环境变量中添加OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = ''


# 以一般客服场景为例
# 在电信公司的客服聊天机器人场景中, 如果用户在对话中先是询问了账单问题, 接着又谈到了网络连接问题,
# ConversationBufferMemory可以用来记住整个与用户的对话历史, 可以帮助AI在回答网络问题时, 还记得账单问题的相关细节, 从而提供更连贯的服务.


memory = ConversationBufferMemory()
memory.save_context({'input': '你好.'}, {'output': '怎么了?'})

variables = memory.load_memory_variables({})
print(variables)

