import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

# Loader
loader = PyPDFLoader(file_path="../docs/essays.pdf")
pages = loader.load()

print(pages[0].page_content)

# Transformers - Splitter
document_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
split_documents = document_splitter.split_documents(documents=pages)

# Embeddings
embeddings = OpenAIEmbeddings()

embedding = embeddings.embed_query(text="This is some sample text.")

print(embedding)

# Vector Store
db = DocArrayInMemorySearch.from_documents(documents=split_documents, embedding=embeddings)

result = db.search(query="Some test search", search_type="similarity")

# Retriever
retriever = db.as_retriever()

# Conversational Retrieval Chain

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo"),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    return_generated_question=True
)
