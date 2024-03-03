import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

openai_api_key = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")

# result = chat.invoke("Hi there, who are you?")
# print(result.pretty_print())
#
# prompt_template = PromptTemplate.from_template(template="Translate to chinese text in a backticks ```{text}```")
#
# prompt = prompt_template.format_prompt(text="Hi there, who are you?")
#
# result1 = chat.invoke(prompt)
# print(result1.content)

response_schema = ResponseSchema(name="test", description="Way how to extract test feature.")

parser = StructuredOutputParser.from_response_schemas([response_schema])
instruction = parser.get_format_instructions()
pass






