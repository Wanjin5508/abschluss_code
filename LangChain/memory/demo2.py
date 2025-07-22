import openai
import os
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI


# 在环境变量中添加OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = ''
openai.api_key = ''


conversation_with_summary = ConversationChain(
    llm=ChatOpenAI(temperature=0, max_tokens=1000), 
    # k=2, 只保留最近 2 轮对话信息在 memory 中
    memory=ConversationBufferWindowMemory(k=2), 
    verbose=True
)



res = conversation_with_summary.predict(input="嗨, 最近一切可好? 请用简体中文回复.")
print(res)
print('--------------------------------------------------------')


res = conversation_with_summary.predict(input="出了什么问题吗? 请用简体中文回复.")
print(res)
print('--------------------------------------------------------')


res = conversation_with_summary.predict(input="现在情况好转了吗? 请用简体中文回复.")
print(res)
print('--------------------------------------------------------')


# 注意: 第一轮对话的信息已经消失不见了.
res = conversation_with_summary.predict(input="那是怎么解决的? 请用简体中文回复.")
print(res)
