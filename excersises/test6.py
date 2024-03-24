import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

chat = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0.6)

prompt_template_string = ("Extract following data from the text delimited by triple backticks:\n"
                          "day_part: What part of the day is described in the text?\n"
                          "whether: What whether is mentioned in the text? Also, if whether is not\n"
                          "directly mentioned then try to make conclusion from context.\n"
                          " If whether is not described at all\n"
                          "respond with null\n"
                          "Given text is: ```{text}```\n"
                          "{format_instructions}")

text = "We are walking during the bright shiny night full of stars. Moon was so big that you can touch it."

prompt_string = prompt_template_string.format(text=text, format_instructions="")

chat_prompt_template = ChatPromptTemplate.from_template(template=prompt_template_string)
chat_prompt = chat_prompt_template.format_prompt(text=text, format_instructions="")

print(chat_prompt)

prompt_template = PromptTemplate.from_template(template=prompt_template_string)
prompt = prompt_template.format_prompt(text=text, format_instructions="")

print(prompt)

day_part_schema = ResponseSchema(name="day_part", description="What part of the day is described in the text?")
whether = ResponseSchema(name="whether", description="What whether is mentioned in the text? Also, if whether is not directly mentioned then try to make conclusion from context. If whether is not described at all respond with null")

output_parser = StructuredOutputParser.from_response_schemas(response_schemas=[day_part_schema, whether])
format_instructions = output_parser.get_format_instructions()

print(format_instructions)

prompt_template_output = PromptTemplate.from_template(template=prompt_template_string)
prompt_output = prompt_template_output.format_prompt(text=text, format_instructions=format_instructions)

print(prompt_output)

response = chat.invoke(prompt_output)

print(response.content)

response_dict = output_parser.parse(response.content)

print(response_dict)




