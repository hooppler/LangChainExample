import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load text
loader = TextLoader(file_path="../resources/docs/short_story.txt")
documents = loader.load()

# Split Documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_documents = text_splitter.split_documents(documents=documents)

# Embeddings
embeddings = OpenAIEmbeddings()

# ChromaDB
db = Chroma.from_documents(documents=split_documents, embedding=embeddings, persist_directory="../resources/chroma_db1")

# Retriever
retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 2})

# Conversational Chain
chat = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")
memory = ConversationBufferMemory()
conversational_chain = ConversationalRetrievalChain.from_llm(
    llm=chat,
    memory=memory,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    return_generated_question=True,
)

# Chat example
chat_history = []

query1 = "Hi, my name is Aleksandar"
# result1 = conversational_chain.invoke({"question": query1, "chat_history": chat_history})
# chat_history.extend([(query1, result1["answer"])])

result1 = conversational_chain.invoke({"question": query1, "chat_history": chat_history})

query2 = "Can you tell me how to write classification essay?"
# result2 = conversational_chain.invoke({"question": query2, "chat_history": chat_history})
# chat_history.extend([(query2, result2["answer"])])

result2 = conversational_chain.invoke({"question": query2, "chat_history": chat_history})

query3 = "Do you remember my name?"
# result3 = conversational_chain.invoke({"question": query3, "chat_history": chat_history})
# chat_history.extend([(query3, result3["answer"])])

result3 = conversational_chain.invoke({"question": query3, "chat_history": chat_history})

print(chat_history)

pass

