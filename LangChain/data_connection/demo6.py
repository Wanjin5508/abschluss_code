from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
import time
import os
import openai
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader


# 在环境变量中添加OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = ''
openai.api_key = ''


# client = OpenAI()

# 加载长文本
raw_documents = TextLoader('../tests/state_of_the_union.txt').load()

# 实例化文本分割器
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)


# 分割文本
documents = text_splitter.split_documents(raw_documents)

'''
print(len(documents))
print(type(documents))
print('------------------------')
print(type(documents[0]))
print('------------------------')
print(documents[0])
print('------------------------')
print(documents[10])
'''


# 将分割后的文本, 使用OpenAI嵌入模型获取嵌入向量, 并存储在Chroma中
db = Chroma.from_documents(documents, OpenAIEmbeddings(model='text-embedding-3-small'))


# 直接使用文本进行相似语义搜索
query = "What did the president say about Ketanji Brown Jackson"


start_1 = time.time()
docs = db.similarity_search(query)
end_1 = time.time()

print('cost time: ', end_1 - start_1)
print(docs[0].page_content)
print('------------------------------------------------------------')


# 先对文本进行embedding向量化嵌入, 再用向量数据库进行搜索
start_2 = time.time()
embedding_vector = OpenAIEmbeddings(model='text-embedding-3-small').embed_query(query)
docs = db.similarity_search_by_vector(embedding_vector)
end_2 = time.time()
print('cost time: ', end_2 - start_2)
print(docs[0].page_content)

