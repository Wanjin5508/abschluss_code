import langchain
import openai
import os
from langchain_openai import OpenAI,ChatOpenAI
from langchain.memory import ConversationEntityMemory


# 在环境变量中添加OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = ''


# 以法律咨询场景为例
# 在法律咨询的场景中, 客户可能会提到特定的案件名称, 相关法律条款或个人信息
# (如"我在去年的交通事故中受了伤, 想了解关于赔偿的法律建议").
# ConversationEntityMemory 可以帮助AI记住这些关键实体和实体关系细节,
# 从而在整个对话过程中提供更准确, 更个性化的法律建议.


llm = ChatOpenAI(model='gpt-4', temperature=0.9)

memory = ConversationEntityMemory(llm=llm)

input1 = {"input": "公众号《AI学堂》的作者是小朱老师."}

memory.load_memory_variables(input1)
memory.save_context(
    input1,
    {"output": "是吗, 这个公众号是干嘛的?"}
)

res = memory.load_memory_variables({"input": "小朱老师是谁?"})
print(res)

