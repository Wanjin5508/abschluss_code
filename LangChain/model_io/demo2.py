import openai
import os
from langchain_openai import ChatOpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate


# 在环境变量中添加OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = ''
openai.api_key = ''


# model = ChatOpenAI(model_name='gpt-4')


# 创建一个输出解析器, 用于处理带逗号分隔的列表输出
output_parser = CommaSeparatedListOutputParser()

# 获取格式化指令, 该指令告诉模型如何格式化其输出
format_instructions = output_parser.get_format_instructions()

# 创建一个提示模板, 它会基于给定的模板和变量来生成提示
prompt = PromptTemplate(
    template="List five {subject}.\n{format_instructions}",         # 模板内容
    input_variables=['subject'],                                    # 输入变量
    partial_variables={'format_instructions': format_instructions}  # 预定义的变量,这里我们传入格式化指令
)

input_prompt = prompt.format(subject='ice cream flavors')
print(input_prompt)

