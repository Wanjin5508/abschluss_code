import os
import openai
from langchain_openai import OpenAI, OpenAIEmbeddings


# 在环境变量中添加OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = ''
openai.api_key = ''


client = OpenAI()

example_input = [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ]


response = client.embeddings.create(
    input=example_input,
    model="text-embedding-3-small"
    # model="text-embedding-3-large"
)

print(response)
print(len(response.data[0].embedding))
# small: 1536
# large: 3072
