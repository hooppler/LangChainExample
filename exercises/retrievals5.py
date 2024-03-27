import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Loading
loader = PyPDFLoader(file_path="../docs/essays.pdf")
pages = loader.load()

print(pages)

# Transformations - Splitters
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(documents=pages)

# Embeddings
embedding = OpenAIEmbeddings()

embd = embedding.embed_query("This is some example text.")
print(embd)

# Vector Store
db = DocArrayInMemorySearch.from_documents(documents=documents, embedding=embedding)

res = db.search(query="What is this?", search_type="similarity")

print(res)

# Retriever
retriever = db.as_retriever()

# Conversational Chain
chat = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")
conversational_chain = ConversationalRetrievalChain.from_llm(
    llm=chat,
    retriever=retriever,
    return_source_documents=True,
    return_generated_question=True,
)
