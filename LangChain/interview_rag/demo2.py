import os
from langchain.memory import ConversationBufferWindowMemory


# 在环境变量中添加OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = ''


# 以商品咨询场景为例
# 在一个电商平台上, 如果用户询问关于特定产品的问题 (如手机的电池续航时间),
# 然后又问到了配送方式, ConversationBufferWindowMemory可以帮助AI只专注于
# 最近的一两个问题 (如配送方式), 而不是整个对话历史, 以提供更快速和专注的答复.

# 只保留最近一轮的对话信息
memory = ConversationBufferWindowMemory(k=1)
print(memory)

