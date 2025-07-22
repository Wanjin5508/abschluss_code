import openai
import os
from langchain_openai import OpenAI, ChatOpenAI
from langchain_community.document_loaders import ArxivLoader


# 在环境变量中添加OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = ''
openai.api_key = ''


# query = "2005.14165"
query = "2412.02612v1"
docs = ArxivLoader(query=query, load_max_docs=5).load()

print(len(docs))
print('-------------------------------------------------')
print(docs[0].metadata)

