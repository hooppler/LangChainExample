import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

openai_api_key = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0.9)

result = chat.invoke("Hi, who are you ?")

print(result.content)


# Old fassion way
prompt_template = "Translate to ancient Egyptian text given in triple backticks. Text: ```{text}```"

text = "This is some ordinary boring text."

prompt = prompt_template.format(text=text)

result1 = chat.invoke(prompt)

print(result1.content)

# New fassion way

prompt_template1 = ChatPromptTemplate.from_template(prompt_template)
prompt = prompt_template1.format_messages(text=text)

response2 = chat.invoke(input=prompt)

print(response2.content)

# Parser Example

prompt_template3_string = "Generate text about given topic {topic}. {format_instructions}"

topic = "Neural Networks"
format_instructions = ""

prompt_template3 = ChatPromptTemplate.from_template(prompt_template3_string)
prompt3 = prompt_template3.format_messages(topic=topic, format_instructions=format_instructions)

response3 = chat.invoke(prompt3)

print(prompt_template3.messages)
print(topic)
print(response3.content)

title_schema = ResponseSchema(name="title", description="Give some appropriate title.")
description_schema = ResponseSchema(name='description', description="Give some description from given text")

parser = StructuredOutputParser.from_response_schemas([title_schema, description_schema])

format_instructions = parser.get_format_instructions()

print(format_instructions)

prompt4 = prompt_template3.format_messages(topic=topic, format_instructions=format_instructions)

print(prompt4)

response4 = chat.invoke(prompt4)

print(response4.content)


output_dict = parser.parse(response4.content)

print("--------------------------------------------")
print(output_dict)








