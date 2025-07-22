import openai
import os
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI


# 在环境变量中添加OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = ''
openai.api_key = ''


llm = ChatOpenAI(temperature=0, max_tokens=1000)

conversation = ConversationChain(
    llm=llm, 
    verbose=True, 
    memory=ConversationBufferMemory()
)


conversation.predict(input='嗨, 你好啊!')

res = conversation.predict(input="我很好, 正在和AI进行对话呢!")

print(res)

res = conversation.predict(input="给我讲讲关于你的事情吧.")
print(res)
