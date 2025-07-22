import openai
import os
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


os.environ['OPENAI_API_KEY'] = ''
openai.api_key = ''


llm = ChatOpenAI(temperature=0, max_tokens=1000)


# AssertionError: Function must have a docstring if description not provided.
# 注意: 下面的tool工具函数, 一定要用3个"""来加上注释说明, 不能用"#"来表示所谓的注释, 否则报错
@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

tools = [get_word_length]


# 创建模版信息
prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant, but bad at calculating lengths of words."), 
                    MessagesPlaceholder("chat_history", optional=True),
                    ("human", "{input}"),
                    MessagesPlaceholder("agent_scratchpad"),
                ]
            )


MEMORY_KEY = "chat_history"
# 创建memory, 参数传入关键词
memory = ConversationBufferMemory(memory_key=MEMORY_KEY, return_messages=True)

# 实例化Agent, 3个参数: 模型llm, 工具tools, 提示prompt
agent = create_openai_functions_agent(llm, tools, prompt)

# 实例化Agent执行器
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

res = agent_executor.invoke({"input": "how many letters in the word educa?"})
res = agent_executor.invoke({"input": "is that a real word?"})
