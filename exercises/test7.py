import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

chat = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0.6)

prompt_template_string = ("Extract following data from text delimited by triple backticks."
                          "part_of_the_day: Conclude what part of the day provided text describe."
                          "weather: Extract information that describe whether in the text."
                          "Provided text is: ```{text}```"
                          "{format_instructions}")
text = "We are walking through the streets of the city, big clock shows 6PM, and wind blown my heat."

prompt_string = prompt_template_string.format(text=text, format_instructions="")

# result = chat.invoke(prompt_string)
#
# print(result)

prompt_template = ChatPromptTemplate.from_template(template=prompt_template_string)
prompt = prompt_template.format_prompt(text=text, format_instructions="")

result = chat.invoke(prompt)

print(result.content)

part_of_the_day_schema = ResponseSchema(name="part_of_the_day", description="Conclude what part of the day provided text describe.")
weather_schema = ResponseSchema(name="weather", description="Extract information that describe whether in the text.")

output_parser = StructuredOutputParser.from_response_schemas(response_schemas=[part_of_the_day_schema, weather_schema])
format_instructions = output_parser.get_format_instructions()

prompt = prompt_template.format_prompt(text=text, format_instructions=format_instructions)

response = chat.invoke(prompt)

print(response.content)

response_dict = output_parser.parse(response.content)

print(response_dict)
