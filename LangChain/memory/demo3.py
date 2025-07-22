import openai
import os
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI


# 在环境变量中添加OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = ''
openai.api_key = ''

llm=ChatOpenAI(temperature=0, max_tokens=1000)


memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=10)
memory.save_context({"input": "hi"}, {"output": "whats up"})
memory.save_context({"input": "not much you"}, {"output": "not much"})

memory.load_memory_variables({})


messages = memory.chat_memory.messages
print(messages)
previous_summary = ""
res = memory.predict_new_summary(messages, previous_summary)
print(res)

