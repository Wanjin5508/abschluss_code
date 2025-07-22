from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
import openai
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


# 在环境变量中添加OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = ''
openai.api_key = ''


# 定义一个提示模板
example_prompt = PromptTemplate(
    input_variables=["input", "output"],            # 输入变量的名字
    template="Input: {input}\nOutput: {output}",    # 实际的模板字符串
)

# 这是一个假设的任务示例列表, 用于创建反义词
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]


# 从给定的示例中创建一个语义相似性选择器
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,                          # 可供选择的示例列表
    OpenAIEmbeddings(model='text-embedding-3-small'),   # 用于生成嵌入向量的嵌入类, 用于衡量语义相似性
    Chroma,                            # 用于存储嵌入向量并进行相似性搜索的VectorStore类
    k=1                                # 要生成的示例数量
)

# 创建一个FewShotPromptTemplate对象
similar_prompt = FewShotPromptTemplate(
    example_selector=example_selector,            # 提供一个ExampleSelector替代示例
    example_prompt=example_prompt,                # 前面定义的提示模板
    prefix="Give the antonym of every input",     # 前缀模板
    suffix="Input: {adjective}\nOutput:",         # 后缀模板
    input_variables=["adjective"],                # 输入变量的名字
)

# 输入是一种感受, 所以应该选择happy/sad相关的示例.
print(similar_prompt.format(adjective="worried"))

# 输入是一种度量, 所以应该选择tall/short相关的示例
print(similar_prompt.format(adjective="long"))
