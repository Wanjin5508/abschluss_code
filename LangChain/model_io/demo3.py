from langchain.output_parsers import DatetimeOutputParser
from langchain.chains import LLMChain
import openai
import os
from langchain_openai import ChatOpenAI, OpenAI
from langchain.prompts import PromptTemplate


# 在环境变量中添加OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = ''
openai.api_key = ''
llm = OpenAI()

output_parser = DatetimeOutputParser()
template = """Answer the users question:

{question}

{format_instructions}"""

prompt = PromptTemplate.from_template(
    template,
    partial_variables={'format_instructions': output_parser.get_format_instructions()},
)

# chain = LLMChain(prompt=prompt, llm=OpenAI())
chain = prompt | llm

output = chain.invoke("around when was bitcoin founded?")
print(output)
print('-------------------------')
print(output_parser.parse(output))

