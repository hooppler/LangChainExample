import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Loader
loader = PyPDFLoader(file_path="../resources/docs/essays.pdf")
documents = loader.load()

# Transform
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=150)
split_documents = text_splitter.split_documents(documents=documents)

# Embedding
embeddings = OpenAIEmbeddings()

# Vector Database (Base class VectorStore)
db = DocArrayInMemorySearch.from_documents(documents=documents, embedding=embeddings)

# Retrievals (Base class VectorStoreRetriever)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
result = db.search(query="Avoid vague language, such as “pretty,” ", search_type="similarity")

# Conversational Retrieval Chain
conversation = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    return_generated_question=True,
)

chat_history = []

query1 = "Hi, my name is Aleksandar"
result1 = conversation.invoke({"question": query1, "chat_history": chat_history})
chat_history.extend([(query1, result1["answer"])])

query2 = "Can you tell me how to write classification essay?"
result2 = conversation.invoke({"question": query2, "chat_history": chat_history})
chat_history.extend([(query2, result2["answer"])])

query3 = "Do you remember my name?"
result3 = conversation.invoke({"question": query3, "chat_history": chat_history})
chat_history.extend([(query3, result3["answer"])])

print(result1["question"])
print(result1["answer"])
print(result2["question"])
print(result2["answer"])
print(result3["question"])
print(result3["answer"])

