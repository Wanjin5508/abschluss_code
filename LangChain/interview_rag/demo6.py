import langchain
import os
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import VectorStoreRetrieverMemory
from langchain_chroma import Chroma
from langchain.chains import ConversationChain


# 在环境变量中添加OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = ''


# 以一般客服场景为例
# 在电信公司的客服聊天机器人场景中, 如果用户在对话中先是询问了账单问题, 接着又谈到了网络连接问题,
# ConversationBufferMemory可以用来记住整个与用户的对话历史, 可以帮助AI在回答网络问题时, 还记得账单问题的相关细节, 从而提供更连贯的服务.


# -----------------------------------------------------------------------


# 以商品咨询场景为例
# 在一个电商平台上, 如果用户询问关于特定产品的问题 (如手机的电池续航时间),
# 然后又问到了配送方式, ConversationBufferWindowMemory可以帮助AI只专注于
# 最近的一两个问题 (如配送方式), 而不是整个对话历史, 以提供更快速和专注的答复.


# -----------------------------------------------------------------------


# 以法律咨询场景为例
# 在法律咨询的场景中, 客户可能会提到特定的案件名称, 相关法律条款或个人信息
# (如"我在去年的交通事故中受了伤, 想了解关于赔偿的法律建议").
# ConversationEntityMemory 可以帮助AI记住这些关键实体和实体关系细节,
# 从而在整个对话过程中提供更准确, 更个性化的法律建议.



# -----------------------------------------------------------------------


# 对历史对话进行阶段性总结摘要, 以教育辅导场景为例:
# 在一系列的教育辅导对话中, 学生可能会提出不同的数学问题或理解难题,
# (如"我不太理解二次方程的求解方法."), ConversationSummaryMemory 可以帮助AI 
# 总结之前的辅导内容和学生的疑问点, 以便在随后的辅导中提供更针对性的解释和练习.

# -----------------------------------------------------------------------

# 需要获取最新的对话, 又要兼顾较早历史对话, 以技术支持场景为例:
# 在处理一个长期的技术问题时(如软件故障排查), 用户可能会在多次对话中提供不同的错误信息和反馈.
# ConversationSummaryBufferMemory 可以帮助AI保留最近几次交互的详细信息,
# 同时提供历史问题处理的摘要, 以便于更有效地识别和解决问题.

# --------------------------------------------------------------------

# 回溯最近和最关键的对话信息, 以金融咨询场景为例:
# 在金融咨询聊天机器人中, 客户可能会提出多个问题, 涉及投资, 市场动态或个人财务规划,
# (如"我想了解股市最近的趋势以及如何分配我的投资组合."),
# ConversationTokenBufferMemory 可以帮助AI聚焦于最近和最关键的几个问题,
# 同时避免由于记忆过多而导致的信息混淆.

# --------------------------------------------------------------------

# 基于向量检索的对话信息, 以了解最新新闻事件为例:
# 用户可能会对特定新闻事件提出问题, 如"最近的经济峰会有什么重要决策?",
# VectorStoreRetrieverMemory 能够快速从大量历史新闻数据中检索出与当前问题最相关的信息,
# 即使这些信息在整个对话历史中不是最新的, 也能提供及时准确的背景信息和详细报道.


llm = ChatOpenAI(model='gpt-4o')

vectorstore = Chroma(embedding_function=OpenAIEmbeddings(model='text-embedding-3-large'))

retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))

memory = VectorStoreRetrieverMemory(retriever=retriever)

memory.save_context({"input": "小李是程序员."}, {"output": "知道了, 小李是程序员."})
memory.save_context({"input": "三胖子是小李的艺名."}, {"output": "明白, 三胖子是小李的艺名."})

memory.save_context({"input": "我喜欢吃火锅."}, {"output": "听起来很好吃."})
memory.save_context({"input": "我喜欢看摔跤比赛."}, {"output": "我也是."})


PROMPT_TEMPLATE = """以下是人类和AI之间的友好对话. AI话语多且提供了许多来自其上
下文的具体细节. 如果AI不知道问题的答案, 它会诚实地说不知道.
以前对话的相关片段:
{history}
(如果不相关, 你不需要使用这些信息)
当前对话:
人类:{input}
AI:
"""

prompt = PromptTemplate(input_variables=["history", "input"], template=PROMPT_TEMPLATE)


conversation_chain = ConversationChain(
    prompt=prompt,
    memory=memory,
    llm=llm,
    verbose=True
)

res = conversation_chain.predict(input="我喜欢的食物是什么?")
print(res)

