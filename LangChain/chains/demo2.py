import openai
import os
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


# 在环境变量中添加OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = ''
openai.api_key = ''


# 这是一个 LLMChain, 用于根据剧目的标题撰写简介
llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.7, max_tokens=1000)

template = """你是一位剧作家。根据戏剧的标题, 你的任务是为该标题写一个简介。

标题: {title}
剧作家: 以下是对上述戏剧的简介: """

prompt = PromptTemplate(input_variables=["title"], template=template)


# 最新版本的langchain, 已经由管道符号代替了早期的LLMChain
# chain = LLMChain(llm=llm, prompt=prompt)
# res = chain.invoke("三体人不是无法战胜的")
# print(res)


chain = prompt | llm
res = chain.invoke("三体人不是无法战胜的")
print(res.content)

