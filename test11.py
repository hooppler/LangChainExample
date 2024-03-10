import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.retrievers import CohereRagRetriever

