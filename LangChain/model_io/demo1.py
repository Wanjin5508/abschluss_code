import openai
import os
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage


# 在环境变量中添加OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = ''
openai.api_key = ''


# chat_model = ChatOpenAI(model_name='gpt-4')

messages = [
    SystemMessage(content='You are a helpful assistant.'),
    HumanMessage(content='Who won the world champion in 2022?'),
    AIMessage(content='The Golden State Warriors won the World Champion in 2022.'),
    HumanMessage(content='Where was it palyed?')
]

# chat_result = chat_model.invoke(messages)

# print(chat_result)
# print('--------------------')
# print(type(chat_result))



# llm = ChatOpenAI(model_name='gpt-3.5-turbo', max_tokens=1024)
llm = ChatOpenAI(model_name='gpt-4')
# llm.max_tokens = 1024
print(llm.max_tokens)
# print(llm('Tell me a joke'))
res = llm.invoke('讲3个给程序员听的笑话, 要幽默诙谐的感觉!')
print(res.content)

