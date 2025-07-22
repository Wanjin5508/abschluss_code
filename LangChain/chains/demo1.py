import openai
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI


# 在环境变量中添加OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = ''
openai.api_key = ''


llm = OpenAI(temperature=0, max_tokens=1000)


prompt = PromptTemplate(
    input_variables=['product'],
    template="给制造{product}的有限公司取10个好名字, 并给出完整的公司名称"
)


# chain = LLMChain(llm=llm, prompt=prompt)
# print(chain.invoke({'product': "性能卓越的GPU"}))

chain = prompt | llm
response = chain.invoke({'product': "性能卓越的GPU"})
print(response)

