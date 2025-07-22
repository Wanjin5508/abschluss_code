import openai
import os
from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
# 这是一个SimpleSequentialChain, 按顺序运行这两个链
# from langchain.chains import SimpleSequentialChain


# 在环境变量中添加OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = ''
openai.api_key = ''


# 这是一个 LLMChain, 用于根据剧目的标题撰写简介
llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.7, max_tokens=1000)

template1 = """你是一位剧作家。根据戏剧的标题, 你的任务是为该标题写一个简介。

标题: {title}
剧作家: 以下是对上述戏剧的简介: """

prompt1 = PromptTemplate(input_variables=["title"], template=template1)


# 这是一个LLMChain, 用于根据剧情简介撰写一篇戏剧评论
template2 = """你是《纽约时报》的戏剧评论家。根据剧情简介, 你的工作是为该剧撰写一篇评论。

剧情简介：
{synopsis}

以下是来自《纽约时报》戏剧评论家对上述剧目的评论："""

prompt2 = PromptTemplate(input_variables=["synopsis"], template=template2)

chain = prompt1 | llm | prompt2 | llm

response = chain.invoke("三体人不是无法战胜的")
print(response.content)
print(response.response_metadata['token_usage'])
