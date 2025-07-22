from langchain.text_splitter import RecursiveCharacterTextSplitter, Language


# 加载待分割长文本
with open('../tests/state_of_the_union.txt') as f:
    state_of_the_union = f.read()


# 定义文本分割器对象
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap  = 20,
    length_function = len,
    add_start_index = True
)


texts = text_splitter.create_documents([state_of_the_union])
print(texts[0])
print('----------------------------')
print(texts[1])
print('****************************')

print(type(texts))
print(len(texts))
print(type(texts[0]))

print('----------------------------')

res = [e.value for e in Language]
print(res)
print(len(res))



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

'''
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
'''
