import openai
import os
from langchain.agents import tool, load_tools, initialize_agent, AgentType
from langchain_openai import ChatOpenAI


# 在环境变量中添加OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = ''
openai.api_key = ''
# 在环境变量中添加SERPAPI_API_KEY
os.environ["SERPAPI_API_KEY"] = ''


llm = ChatOpenAI(model="gpt-4", temperature=0, max_tokens=1000)


# 使用tools, 这里定义llm是因为工具要用到llm
# llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# 初始化agnet对象, 传入工具tools, 大模型llm, 提示词模板prompt
agent = initialize_agent(tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# agent.run("姚明的老婆的身高是多少?")
# agent.invoke({"input": "姚明的老婆的身高是多少?"})
# agent.invoke("姚明的老婆的身高是多少?")
agent.invoke("谁是莱昂纳多·迪卡普里奥的女朋友?她现在年龄的0.43次方是多少?")
