import openai
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter


os.environ['OPENAI_API_KEY'] = ''
openai.api_key = ''


with open('real_estate_sales_data.txt') as f:
    real_estate_sales = f.read()

print(len(real_estate_sales))
# 4215
# print(real_estate_sales[:10])



# 构造字符文本切分器对象
text_splitter = CharacterTextSplitter(        
    separator = r'\d+\.',
    chunk_size = 100,
    chunk_overlap  = 0,
    length_function = len,
    is_separator_regex = True
)


# 切分数据文档, 组装成Document格式
docs = text_splitter.create_documents([real_estate_sales])


# print(len(docs))
# 70
# print('---------------')
# print(docs[0])
# print('---------------')
# print(docs[10])


'''
# 实例化faiss向量数据库
db = FAISS.from_documents(docs, OpenAIEmbeddings(model='text-embedding-3-large'))


# 模拟一个小案例
query = '小区吵不吵?'
answer_list = db.similarity_search(query)

print('--------------------------------------')
for ans in answer_list:
    print(ans.page_content + '\n')

print('--------------------------------------')

# 存储成本地的向量数据库, 以备后续项目查询使用
db.save_local('real_estates_sale')


# 使用retriever从向量数据库中获取结果, 设置k=3, 提取最相似的3个
topK_retriever = db.as_retriever(search_kwargs={'k': 2})

# 使用similarity_score_threshold设置阈值, 提升结果的相关性质量, 只有相关性超过0.8才能选中
# retriever = db.as_retriever(
#     search_type='similarity_score_threshold',
#     search_kwargs={'score_threshold': 0.7}
# )

print('-----------------------------------')


query = '你们有没有1000万的豪宅啊?'
docs = topK_retriever.invoke(query)

for doc in docs:
    print(doc.page_content + '\n')
'''

