import openai
import os
from langchain_community.document_loaders import TextLoader


docs = TextLoader('../tests/state_of_the_union.txt').load()

print(len(docs))
print(type(docs[0]))

print(docs[0].page_content[:100])

