import openai
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS


os.environ['OPENAI_API_KEY'] = ''
openai.api_key = ''


# 加载已经生成好的faiss向量数据库
db = FAISS.load_local('real_estates_sale', OpenAIEmbeddings(model='text-embedding-3-large'), allow_dangerous_deserialization=True)


llm = ChatOpenAI(model_name='gpt-4o')

# query = '你们有没有1000万的豪宅啊?'
# query = '小区吵不吵?'

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever(search_kwargs={'k': 2})
)


# docs = qa_chain.invoke({'query': query})
# print(docs)


query = '你们这里可以贷款买房吗?'
res = qa_chain.invoke({'query': query})
print(res)
