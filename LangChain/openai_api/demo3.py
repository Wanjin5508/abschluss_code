import pandas as pd
import tiktoken
import ast
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
# * TSNE是一种用于数据可视化的降维方法, 尤其擅长处理高维数据的可视化
from sklearn.manifold import TSNE
from openai import OpenAI
import os
os.environ['OPENAI_API_KEY'] = ''


input_datapath = "./data/fine_food_reviews_1k.csv"
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




'''
# 设置要筛选的评论数量为1000
top_n = 1000
# 对DataFrame进行排序, 基于"Time"列, 然后选取最后的2000条评论
# 这个假设是, 我们认为最近的评论可能更相关, 因此我们将对它们进行初始筛选
df = df.sort_values("Time").tail(top_n * 2) 
# 丢弃"Time"列, 因为我们在这个分析中不再需要它
df.drop("Time", axis=1, inplace=True)
# 从'embedding_encoding'获取编码
encoding = tiktoken.get_encoding(embedding_encoding)

# 计算每条评论的token数量, 我们通过使用encoding.encode方法获取每条评论的token数, 然后把结果存储在新的'n_tokens'列中
df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))

# 如果评论的token数量超过最大允许的token数量, 我们将忽略(删除)该评论
# 我们使用.tail方法获取token数量在允许范围内的最后top_n(1000)条评论
df = df[df.n_tokens <= max_tokens].tail(top_n)

# 打印出剩余评论的数量
# print(len(df))
'''

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


# ! 首先确保你的嵌入向量都是等长的
assert df_embedded['embedding_vec'].apply(len).nunique() == 1


# * 将嵌入向量列表转换为二维 numpy 数组
matrix = np.vstack(df_embedded['embedding_vec'].values)


# * 创建一个t-SNE模型 --> 传统机器学习, 用于降维+聚类
# n_components: 表示降维后的维度(在这里是2D), 也可以用于 3d 可视化, 只需要改为3
# perplexity: 可以被理解为近邻的数量
# random_state: 是随机数生成器的种子
# init: 设置初始化方式
# learning_rate: 学习率
tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)


# * 使用t-SNE对数据进行降维, 得到每个数据点在新的2D空间中的坐标
vis_dims = tsne.fit_transform(matrix)

# 定义了五种不同的颜色, 用于在可视化中表示不同的等级
colors = ["red", "darkorange", "gold", "turquoise", "darkgreen"]


# 从降维后的坐标中分别获取所有数据点的横坐标和纵坐标
x = [x for x,y in vis_dims]
y = [y for x,y in vis_dims]

# ! 根据数据点的评分(减1是因为评分是从1开始的, 而颜色索引是从0开始的)获取对应的颜色索引
color_indices = df_embedded.Score.values - 1

# 确保你的数据点和颜色索引的数量匹配
assert len(vis_dims) == len(df_embedded.Score.values)

# 创建一个基于预定义颜色的颜色映射对象
colormap = matplotlib.colors.ListedColormap(colors)
# 使用 matplotlib 创建散点图, 其中颜色由颜色映射对象和颜色索引共同决定, alpha 是点的透明度
plt.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.3)

# 为图形添加标题
plt.title("Amazon ratings using t-SNE")
plt.show()



# * 聚类算法
# 定义要生成的聚类数
n_clusters = 4

# n_clusters: 参数指定了要创建的聚类的数量
# init: 参数指定了初始化方法(在这种情况下是'k-means++')
# random_state: 参数为随机数生成器设定了种子值, 用于生成初始聚类中心
# n_init=10: 消除警告 'FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4'
kmeans = KMeans(n_clusters = n_clusters, init='k-means++', random_state=42, n_init=10)

# 使用matrix来训练KMeans模型
kmeans.fit(matrix)

# kmeans.labels_ 属性包含每个输入数据点所属的聚类的索引
# 我们创建一个新的'Cluster'列, 在这个列中每个数据点都被赋予其所属的聚类的标签
df_embedded['Cluster'] = kmeans.labels_

# 首先为每个聚类定义一个颜色
colors = ["red", "green", "blue", "purple"]

# 使用t-SNE来降维数据, 这里我们只考虑'embedding_vec'列
tsne_model = TSNE(n_components=2, random_state=42)
vis_data = tsne_model.fit_transform(matrix)

# 可以从降维后的数据中获取x和y坐标
x = vis_data[:, 0]
y = vis_data[:, 1]

# 'Cluster'列中的值将被用作颜色索引
color_indices = df_embedded['Cluster'].values

# 创建一个基于预定义颜色的颜色映射对象
colormap = matplotlib.colors.ListedColormap(colors)

# 使用 matplotlib 创建散点图, 其中颜色由颜色映射对象和颜色索引共同决定
plt.scatter(x, y, c=color_indices, cmap=colormap)

# 为图形添加标题
plt.title("Clustering visualized in 2D using t-SNE")

# 显示图形
plt.show()
