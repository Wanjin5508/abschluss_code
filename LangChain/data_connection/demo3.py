import openai
import os
from langchain_openai import OpenAI
from langchain_community.document_loaders import UnstructuredURLLoader


# 在环境变量中添加OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = ''
openai.api_key = ''

urls = ["https://react-lm.github.io/"]
# urls = ["https://platform.openai.com/docs/guides/embeddings/what-are-embeddings"]

data = UnstructuredURLLoader(urls=urls).load()

print(data)
print('---------------------------')
print(data[0].page_content)
