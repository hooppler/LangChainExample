import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

openai_api_key = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)

response = chat.invoke("Hallo my dear, who the hell are you?")

print(response.content)

template_sting = "Translate to French text delimited by triple backticks: ```{text}```"

prompt_template = ChatPromptTemplate.from_template(template_sting)

text = "This is some annoying boring text."

prompt = prompt_template.format_messages(text=text)

response1 = chat.invoke(prompt)

print(response1.content)

