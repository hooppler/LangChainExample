import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

openai_api_key = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0.5)

prompt_template_string = ("Extract following information from given text delimited by "
                          "triple backticks."
                          "part_of_the_day: What part of the day is described in the text."
                          "whether: What is the weather described in the text."
                          "text: ```{text}```"
                          "{structured_output}")

text = "We was walking through the warm and dark night. It was slowly rain at the wind."

part_of_the_day_schema = ResponseSchema(name="part_of_the_day", description="What part of the day is described in the text.")
whether = ResponseSchema(name="whether", description="What is the weather described in the text.")

parser = StructuredOutputParser.from_response_schemas([part_of_the_day_schema, whether])
structured_output = parser.get_format_instructions()

prompt_template = ChatPromptTemplate.from_template(template=prompt_template_string)

prompt = prompt_template.format_prompt(text=text, structured_output=structured_output)

response = chat.invoke(prompt)

res = parser.parse(response.content)

print(res)

pass



