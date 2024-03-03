import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

openai_api_key = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")

result = chat.invoke("Hi there, who are you?")

print(result.content)

prompt_template_string1 = "Translate to Italian text delimited by triple backticks. ```{text}```"
prompt_template_string2 = "Summarise provided text"

prompt_template = ChatPromptTemplate.from_messages(messages=[prompt_template_string1, prompt_template_string2])

print(prompt_template.messages)

text = "This is an preaty simple and stupid sentence that you have to translate."

prompt = prompt_template.format_messages(text=text)

print(prompt[0].content)

result = chat.invoke(input=prompt)

print(result.content)
