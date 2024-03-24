import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings




loader = PyPDFLoader(file_path="docs/essays.pdf")

pages = loader.load()

print(pages[0].page_content)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

documents = text_splitter.split_documents(pages)

print("-----------------------------------")
print(len(documents))

embeddings = OpenAIEmbeddings()

embedding = embeddings.embed_documents(documents)

pass
