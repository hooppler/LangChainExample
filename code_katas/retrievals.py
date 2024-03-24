import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch

# Load
loader = PyPDFLoader(file_path="docs/essays.pdf")
docs = loader.load()

# Transform
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
split_documents = text_splitter.split_documents(docs)

# Embeddings
embeddings = OpenAIEmbeddings()
embedding = embeddings.embed_documents([split_documents[0].page_content])

# Vector Databases
db = DocArrayInMemorySearch.from_documents(split_documents, embeddings)

# Retrievers
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
