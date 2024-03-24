import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

# Loader
loader = PyPDFLoader(file_path="../docs/essays.pdf")
documents = loader.load()

# Text Splitters
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
split_documents = text_splitter.split_documents(documents)

# Embeddings
embedding = OpenAIEmbeddings()
embedding1 = embedding.embed_query("Some test text.")

print(embedding1)

# Vector Store
db = DocArrayInMemorySearch.from_documents(documents=documents, embedding=embedding)
result = db.search(query="What is the most important?", search_type="similarity")

# Retrievals
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# ConversationalRetrieverChain
conversation = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    return_generated_question=True
)
