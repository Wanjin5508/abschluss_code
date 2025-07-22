import os
from langchain_openai import ChatOpenAI
from langchain_community.memory.kg import ConversationKGMemory


# 在环境变量中添加OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = ''


# 以医疗咨询场景为例
# 在医疗咨询中, 一个病人可能会描述多个症状和过去的医疗历史,
# (如"我有糖尿病史, 最近觉得经常口渴和疲劳."),
# ConversationKGMemory 可以构建一个包含病人症状, 疾病历史和可能的健康关联的知识图谱,
# 从而帮助AI提供更全面和深入的医疗建议.


llm = ChatOpenAI(model='gpt-4o')

memory = ConversationKGMemory(llm=llm)


memory.save_context({"input": "小李是糖尿病患者."}, {"output": "知道了, 小李有糖尿病."})
memory.save_context({"input": "小李说他有糖尿病史, 最近觉得经常口渴和疲劳."}, {"output": "明白, 小李有糖尿病史, 最近觉得经常口渴和疲劳."})
memory.save_context({"input": "三胖子是小李的艺名."}, {"output": "明白, 三胖子是小李的艺名."})

res = memory.load_memory_variables({"input": "告诉我关于三胖子的信息."})
print(res)

res = memory.load_memory_variables({"input": "小李有什么症状?"})
print(res)
