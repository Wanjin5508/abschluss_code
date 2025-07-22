import openai
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import gradio as gr
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS


os.environ['OPENAI_API_KEY'] = ''
openai.api_key = ''


def initialize_sales_bot(vector_store_dir: str='real_estates_sale'):
    # 加载已经构建好的faiss向量数据库
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(model='text-embedding-3-large'), allow_dangerous_deserialization=True)
    
    # 实例化LLM对象, 采用GPT-4o-mini模型
    llm = ChatOpenAI(model_name='gpt-4o', temperature=0)
    
    # 将销售顾问设置为全局变量
    global SALES_BOT
    # 带召回检索功能的LLM Chain
    SALES_BOT = RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever(search_kwargs={'k': 2})
    )

    return SALES_BOT


# 销售顾问机器人对话函数
def sales_chat(message, history):
    # print(f"[message]{message}")
    # print(f"[history]{history}")

    # 允许和ChatGPT大模型聊天
    enable_chat = True
    
    # 调用销售顾问机器人, 传入询问message, 获取LLM Chain的返回信息ans
    ans = SALES_BOT.invoke({'query': message})

    # 如果检索出结果, 或者开了大模型聊天模式
    # 返回RetrievalQA combine_documents_chain整合的结果
    if ans or enable_chat:
        return ans['result']
    # 否则输出套路话术
    else:
        return '这个问题我要问问领导.'
    

def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title='AI房产销售顾问',
        chatbot=gr.Chatbot(height=500)
    )

    demo.launch(share=True, server_name='0.0.0.0')


if __name__ == '__main__':
    # 初始化房产销售机器人
    initialize_sales_bot()
    
    # 启动Gradio服务
    launch_gradio()
