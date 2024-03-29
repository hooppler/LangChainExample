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
pages = loader.load()

# Transform - splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(documents=pages)

# Embeddings
embeddings = OpenAIEmbeddings()

embd = embeddings.embed_query("This is some example text.")
print(embd)

# Vector Store
db = DocArrayInMemorySearch.from_documents(documents=documents, embedding=embeddings)

result = db.search(query="Some test text.", search_type="similarity")

print(result)


# Retriever
retriever = db.as_retriever()

# Conversational Chain
chat = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=chat,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    return_generated_question=True
)



