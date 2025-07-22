import openai
import os
from langchain_openai import ChatOpenAI, OpenAI
from langchain.prompts import PromptTemplate


# 在环境变量中添加OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = ''
openai.api_key = ''


model = ChatOpenAI(model_name='gpt-4')

'''
prompt_template = PromptTemplate.from_template(
    "Tell me a {adjective} joke about {content}"
    )

prompt = prompt_template.format(adjective='funny', content='chickens')

print(prompt)
print('---------------------------------')
print(prompt_template)
print('---------------------------------')

res = model.invoke(prompt)
print(res.content)
print(res.usage_metadata['total_tokens'])
print('******')
print(res)
'''


sort_prompt_template = PromptTemplate.from_template("生成可执行的快速排序 {programming_language} 代码")

llm = OpenAI(model='gpt-3.5-turbo-instruct', max_tokens=4000)
# print(llm.invoke(sort_prompt_template.format(programming_language="python")))
print(llm.invoke(sort_prompt_template.format(programming_language="C++")))

