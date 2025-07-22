from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

from langchain_community.vectorstores import Chroma
import time
import os
import openai
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter


# 在环境变量中添加OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = ''
openai.api_key = ''


# client = OpenAI()


html_text = """
<!DOCTYPE html>
<html>
    <head>
        <title>🦜️🔗 LangChain</title>
        <style>
            body {
                font-family: Arial, sans-serif;
            }
            h1 {
                color: darkblue;
            }
        </style>
    </head>
    <body>
        <div>
            <h1>🦜️🔗 LangChain</h1>
            <p>⚡ Building applications with LLMs through composability ⚡</p>
        </div>
        <div>
            As an open source project in a rapidly developing field, we are extremely open to contributions.
        </div>
    </body>
</html>
"""


html_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.HTML,
    chunk_size=60,
    chunk_overlap=0
)


html_docs = html_splitter.create_documents([html_text])


print(len(html_docs))
print(type(html_docs))
print('------------------------------------------')

print(html_docs[0])
print(type(html_docs[0]))
print('------------------------------------------')

print(html_docs)

