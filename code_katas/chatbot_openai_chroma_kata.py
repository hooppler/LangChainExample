import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")

# Loader
loader = PyPDFLoader(file_path="../resources/docs/essays.pdf")
documents = loader.load()

# Transform
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=150)
split_documents = text_splitter.split_documents(documents=documents)

# Embedding
embeddings = OpenAIEmbeddings()

# Vector Store - ChromaDB save to disk
db1 = Chroma.from_documents(split_documents, embeddings, persist_directory="../resources/chroma_db")

# Vector Store - ChromaDB load from disk as new DB
db2 = Chroma(persist_directory="../resources/chroma_db", embedding_function=embeddings)

# Query databases
query = "Avoid vague language, such as “pretty,” "

docs1 = db1.similarity_search(query)
docs2 = db2.similarity_search(query)

print(docs1[0].page_content)
print(docs2[0].page_content)
