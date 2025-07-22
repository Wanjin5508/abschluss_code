import openai
import os
from langchain_openai import ChatOpenAI
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from operator import itemgetter
from typing import Literal
from typing_extensions import TypedDict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


# 在环境变量中添加OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = ''
openai.api_key = ''


# 这是一个 LLMChain, 用于根据剧目的标题和时代的设定, 来撰写简介
llm = ChatOpenAI(model='gpt-4o-mini')


physics_template = """你是一位非常聪明的物理教授。
你擅长以简洁易懂的方式回答关于物理的问题。
当你不知道某个问题的答案时，你会坦诚承认。

这是一个问题:
{query}"""


math_template = """你是一位很棒的数学家。你擅长回答数学问题。
之所以如此出色，是因为你能够将难题分解成各个组成部分，
先回答这些组成部分，然后再将它们整合起来回答更广泛的问题。

这是一个问题:
{query}"""


computer_template = """你是一位顶尖的计算机专家。你擅长回答程序问题。
之所以如此出色，是因为你掌握丰富的算法和模型知识，拥有出色的编程能力，
在回答这些问题时，你可以做到通俗易懂，简明扼要。

这是一个问题:
{query}"""



prompt_1 = ChatPromptTemplate.from_messages(
    [
        ("system", physics_template),
        ("human", "{query}"),
    ]
)
prompt_2 = ChatPromptTemplate.from_messages(
    [
        ("system", math_template),
        ("human", "{query}"),
    ]
)
prompt_3 = ChatPromptTemplate.from_messages(
    [
        ("system", computer_template),
        ("human", "{query}"),
    ]
)


chain_1 = prompt_1 | llm | StrOutputParser()
chain_2 = prompt_2 | llm | StrOutputParser()
chain_3 = prompt_3 | llm | StrOutputParser()

route_system = "将用户的查询请求路由到物理教授, 或者数学家, 或者计算机专家那里去."
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", route_system),
        ("human", "{query}"),
    ]
)


class RouteQuery(TypedDict):
    destination: Literal["物理教授", "数学家", "计算机专家"]


route_chain = (
    route_prompt
    | llm.with_structured_output(RouteQuery)
    | itemgetter("destination")
)

chain = {
    "destination": route_chain,  # "物理教授" or "数学家" or "计算机专家"
    "query": lambda x: x["query"],  # 传入用户的查询请求 query
} | RunnableLambda(
    # if 物理教授, chain_1, 如果数学家, chain_2, 如果计算机专家, chain_3
    lambda x: chain_1 if x["destination"] == "物理教授" else (chain_2 if x["destination"] == "数学家" else chain_3),
)

# print(chain.invoke({"query": "黑体辐射是什么?"}))

# print(chain.invoke({"query": "大于40的第一个质数是多少, 使得这个质数加一能被3整除?"}))

# print(chain.invoke({"query": "请帮我写一个二分查找的代码."}))




# --------------------------------------------------------
# --------------------------------------------------------
# --------------------------------------------------------





biology_template = """你是一位出色的生物学家。你擅长回答生物学问题。
你擅长以通俗易懂的方式回答生物领域的问题，
当你回答问题时, 尽量少用晦涩的名词, 无法回答时, 你也会坦诚承认。

这是一个问题:
{input}"""

# print(chain.invoke("北极熊每年有多长时间冬眠? 每年什么时候生小熊?"))

chinese_template = """你是一位很棒的汉语言文学老师。你擅长回答汉语言文学的问题。
你会用通俗易懂的语言来解释汉语言文学的相关问题，
如果遇到文言文, 你也会用更加容易理解的文言文来回答问题。
当你不知道某个问题的答案时，你会坦诚承认。

这是一个问题:
{input}"""

# print(chain.invoke("小篆这种书法兴盛于什么时候? 被哪种字体取代了?"))
