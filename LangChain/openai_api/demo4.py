import pandas as pd
import time
import ast
import numpy as np
from openai import OpenAI
import os
os.environ['OPENAI_API_KEY'] = ''



input_datapath = "data/fine_food_reviews_1k.csv"
df = pd.read_csv(input_datapath, index_col=0)
df = df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]]
df = df.dropna()


# 将 "Summary" 和 "Text" 字段组合成新的字段 "combined"
df["combined"] = ("Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip())
# print(df.head(2))
# print('------------------------------')
# print(df["combined"])


# 模型类型
# 建议使用官方推荐的第二代嵌入模型: text-embedding-ada-002 (dim=1536, length=8192)
# 建议使用官方推荐的第三代嵌入模型: text-embedding-3-small (dim=1536, length=8192)
# 建议使用官方推荐的第三代嵌入模型: text-embedding-3-large (dim=3072, length=8192)
embedding_model = "text-embedding-ada-002"
# text-embedding-ada-002 模型对应的分词器 (TOKENIZER)
embedding_encoding = "cl100k_base"
# text-embedding-ada-002 模型支持的输入最大 Token 数是8191，向量维度 1536
# 在我们的 DEMO 中过滤 Token 超过 8000 的文本
max_tokens = 8000  


# 直接访问OpenAI的方式, GPT-4o
client = OpenAI(api_key='')


# 设置要筛选的评论数量为1000
top_n = 1000
# 对DataFrame进行排序, 基于"Time"列, 然后选取最后的2000条评论。
# 这个假设是，我们认为最近的评论可能更相关，因此我们将对它们进行初始筛选。
df = df.sort_values("Time").tail(top_n * 2) 
# 丢弃"Time"列，因为我们在这个分析中不再需要它。
df.drop("Time", axis=1, inplace=True)
# 从'embedding_encoding'获取编码
encoding = tiktoken.get_encoding(embedding_encoding)

# 计算每条评论的token数量。我们通过使用encoding.encode方法获取每条评论的token数, 然后把结果存储在新的'n_tokens'列中。
df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))

# 如果评论的token数量超过最大允许的token数量, 我们将忽略(删除)该评论。
# 我们使用.tail方法获取token数量在允许范围内的最后top_n(1000)条评论。
df = df[df.n_tokens <= max_tokens].tail(top_n)

# 打印出剩余评论的数量。
# print(len(df))

# 实际生成会耗时几分钟
# 提醒: 非必须步骤, 可直接复用项目中的嵌入文件 fine_food_reviews_with_embeddings_1k
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

# 对"combined"列中每条评论应用get_embedding函数, 获取相应的嵌入, 结果存储在"embedding"列中
# df["embedding"] = df.combined.apply(lambda x: get_embedding(x, engine=embedding_model))
# df.to_csv("data/fine_food_reviews_with_embeddings_1k.csv")

embedding_datapath = "data/fine_food_reviews_with_embeddings_1k.csv"

df_embedded = pd.read_csv(embedding_datapath, index_col=0)

# print(df_embedded["embedding"])

# print(len(df_embedded["embedding"][0]))
# print(type(df_embedded["embedding"][0]))


# 将字符串转换为向量
df_embedded["embedding_vec"] = df_embedded["embedding"].apply(ast.literal_eval)
# print(len(df_embedded["embedding_vec"][0]))
# print(type(df_embedded["embedding_vec"]))


# 首先确保你的嵌入向量都是等长的
assert df_embedded['embedding_vec'].apply(len).nunique() == 1

# 将嵌入向量列表转换为二维 numpy 数组
# matrix = np.vstack(df_embedded['embedding_vec'].values)


# 计算余弦相似度的函数 --> 按照公式计算
def cosine_similarity(embed1, embed2):
    cosine_sim = np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))
    return cosine_sim


# * 定义一个名为search_reviews的函数
# Pandas DataFrame 产品描述, 数量, 以及一个pprint标志(默认值为True)
def search_reviews(df, product_description, n=3, pprint=True):
    product_embedding = get_embedding(product_description, model="text-embedding-ada-002")
    
    df["similarity"] = df.embedding_vec.apply(lambda x: cosine_similarity(x, product_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)  # 获取最相似的n条评论
        .combined.str.replace("Title: ", "")
        .str.replace("; Content:", ": ")
    )

    if pprint:
        for r in results:
            print(r[:200])
            print()

    return results


# 使用'delicious beans'作为产品描述, 查找与给定产品描述最相似的前3条评论
t1 = time.time()
# res = search_reviews(df_embedded, 'delicious beans', n=3)
res = search_reviews(df_embedded, "bad Cream", n=5)
t2 = time.time()
print(res)
print(t2 - t1)

