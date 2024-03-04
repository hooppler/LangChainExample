import os

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers import ResponseSchema
from langchain.prompts import ChatPromptTemplate

chat = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0.6)

prompt_template_string = ("From the text delimited by triple backticks extract following data:"
                          "part_of_the_day: What part of the day text is talking about."
                          "whether: What whether conditions are described in the text."
                          "Provided text is: {text}"
                          "{format_instructions}")
text = "We where walking through dark city, while our faces were wet because something was in the air."
format_instructions = "Provide results in json format"

result = chat.invoke(prompt_template_string.format(text=text, format_instructions=format_instructions))

print(result.content)

part_of_the_day_schema = ResponseSchema(name="part_of_the_day", description="What part of the day text is talking about.")
whether_schema = ResponseSchema(name="whether", description="What whether conditions are described in the text.")
output_parser = StructuredOutputParser.from_response_schemas(response_schemas=[part_of_the_day_schema, whether_schema])
format_instructions = output_parser.get_format_instructions()

prompt_template = ChatPromptTemplate.from_template(prompt_template_string)
prompt = prompt_template.format_prompt(text=text, format_instructions=format_instructions)

result1 = chat.invoke(prompt)

print(result1.content)

result1_dict = output_parser.parse(result1.content)

print(result1_dict)


















