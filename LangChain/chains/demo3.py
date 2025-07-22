import openai
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain.chains import SequentialChain


# 在环境变量中添加OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = ''
openai.api_key = ''


# 这是一个 LLMChain, 用于根据剧目的标题和时代的设定, 来撰写简介
llm = OpenAI(temperature=0.7, max_tokens=1000)


template = """你是一位剧作家。根据戏剧的标题和设定的时代，你的任务是为该标题写一个简介。

标题: {title}
时代: {era}
剧作家: 以下是对上述戏剧的简介: """

prompt_1 = PromptTemplate(input_variables=["title", "era"], template=template)
synopsis_chain = LLMChain(llm=llm, prompt=prompt_1, output_key="synopsis", verbose=True)


# 这是一个LLMChain, 用于根据剧情简介撰写一篇戏剧评论。

template = """你是《纽约时报》的戏剧评论家。根据该剧的剧情简介，你需要撰写一篇关于该剧的评论。

剧情简介:
{synopsis}

来自《纽约时报》戏剧评论家对上述剧目的评价: """

prompt_2 = PromptTemplate(input_variables=["synopsis"], template=template)
review_chain = LLMChain(llm=llm, prompt=prompt_2, output_key="review", verbose=True)


overall_chain = SequentialChain(
    chains=[synopsis_chain, review_chain],
    input_variables=["era", "title"],
    output_variables=["synopsis", "review"],
    verbose=True)


res = overall_chain.invoke({"title":"三体人不是无法战胜的", "era": "二十一世纪的新中国"})
print(res['review'])

