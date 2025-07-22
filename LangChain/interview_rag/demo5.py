import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from langchain_chroma import Chroma
from operator import itemgetter
from typing import List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.runnables import (
    RunnableLambda,
    ConfigurableFieldSpec,
    RunnablePassthrough,
)
from langchain_core.runnables.history import RunnableWithMessageHistory


# 在环境变量中添加OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = ''


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



class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []


store = {}

def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]



history = get_by_session_id("1")
history.add_message(AIMessage(content="你好啊"))
print(store)



prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个非常得力的助手, 擅长于解决{ability}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

chain = prompt | ChatOpenAI(model="gpt-4")

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_by_session_id,
    input_messages_key="question",
    history_messages_key="history",
)

res = chain_with_history.invoke(
    {
        "ability": "数学",
        "question": "余弦函数是什么意思?"
    },
    config={
        "configurable": {"session_id": "100"}
    }
)

print(res.content)



'''
store = {}

def get_session_history(
    user_id: str, conversation_id: str
) -> BaseChatMessageHistory:
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = InMemoryHistory()
    return store[(user_id, conversation_id)]


prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个非常得力的助手, 擅长于解决{ability}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])


chain = prompt | ChatOpenAI(model="gpt-4")

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="用户的唯一标识符.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="对话的唯一标识符.",
            default="",
            is_shared=True,
        ),
    ],
)


res = with_message_history.invoke(
    {"ability": "日常生活问题", "question": "三胖子是小李的艺名, 而小李是程序员."},
    config={"configurable": {"user_id": "123", "conversation_id": "1"}}
)
print(res.content)
print('-' * 60)

res = with_message_history.invoke(
    {"ability": "日常生活问题", "question": "小李喜欢吃四川火锅."},
    config={"configurable": {"user_id": "123", "conversation_id": "1"}}
)
print(res.content)
print('-' * 60)

res = with_message_history.invoke(
    {"ability": "日常生活问题", "question": "小李喜欢看摔跤比赛.."},
    config={"configurable": {"user_id": "123", "conversation_id": "1"}}
)
print(res.content)
print('-' * 60)

res = with_message_history.invoke(
    {"ability": "日常生活问题", "question": "小李喜欢什么运动和美食呢?"},
    config={"configurable": {"user_id": "123", "conversation_id": "1"}}
)
print(res.content)
print('-' * 60)
'''

