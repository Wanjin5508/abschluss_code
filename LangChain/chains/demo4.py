import openai
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI
# 导入相关的chains模块下的类
from langchain.chains import TransformChain, LLMChain, SimpleSequentialChain


# 在环境变量中添加OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = ''
openai.api_key = ''


# 这是一个 LLMChain, 用于根据剧目的标题和时代的设定, 来撰写简介
llm = OpenAI(temperature=0.7, max_tokens=1000)


with open("../tests/the_old_man_and_the_sea.txt") as f:
    novel_text = f.read()

print(len(novel_text))

# -------------------------------------------------------------------


# 定义一个转换函数, 输入是一个字典, 输出也是一个字典.
def transform_func(inputs: dict) -> dict:
    # 从输入字典中获取"text"键对应的文本.
    text = inputs["text"]

    # 使用split方法将文本按照"\n\n"分隔为多个段落, 并只取前三个, 然后再使用"\n\n"将其连接起来.
    shortened_text = "\n\n".join(text.split("\n\n")[:3])

    # 返回裁剪后的文本, 用"output_text"作为键.
    return {"output_text": shortened_text}


# 使用上述转换函数创建一个TransformChain对象.
# 定义输入变量为["text"], 输出变量为["output_text"], 并指定转换函数为transform_func.
transform_chain = TransformChain(
                                    input_variables=["text"],
                                    output_variables=["output_text"],
                                    transform=transform_func
                                )

# transformed_novel = transform_chain.invoke(novel_text)

'''
print(type(transformed_novel))
print('**********************')
print(transformed_novel.keys())
print('------------------------------------------------------')
print(len(transformed_novel['text']))
print(len(transformed_novel['output_text']))
'''


# ----------------------------------------------------------------



template = """总结下面文本:

{output_text}

总结:"""

prompt = PromptTemplate(input_variables=["output_text"], template=template)
llm_chain = LLMChain(llm=OpenAI(), prompt=prompt, verbose=True)


sequential_chain = SimpleSequentialChain(chains=[transform_chain, llm_chain])

res = sequential_chain.invoke(novel_text[:5000])
print(res)


