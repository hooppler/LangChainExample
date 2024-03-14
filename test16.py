import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers import ResponseSchema
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Prompt strings
prompt_template_string = """
From the text delimited by triple backticks extract following data:
part_of_the_day: Part of the day that text describes.
weather: Weather that is described in the text.
Given text is: ```{text}```
{format_instructions}
"""

format_instructions = "Response should be in the json format."

text = "We are walking through shiny sand at the desert while while sweating without any shadow."


# OpenAI Chat
client = OpenAI(api_key=openai_api_key)

prompt = prompt_template_string.format(text=text, format_instructions=format_instructions)

result = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="gpt-3.5-turbo")
print(result.choices[0].message.content)

# LangChain Text Message
chat = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")

result = chat.invoke(prompt)
print(result.content)

# LangChain Prompt object
prompt_template = ChatPromptTemplate.from_template(template=prompt_template_string)
prompt = prompt_template.format_prompt(text=text, format_instructions=format_instructions)

result = chat.invoke(prompt)
print(result.content)

# LangChain Prompt with Output Parser
part_of_the_day_schema = ResponseSchema(name="part_of_the_day", description="Part of the day that text describes.")
weather_schema = ResponseSchema(name="weather", description="Weather that is described in the text.")
output_parser = StructuredOutputParser.from_response_schemas(response_schemas=[part_of_the_day_schema, weather_schema])

format_instructions = output_parser.get_format_instructions()

prompt = prompt_template.format_prompt(text=text, format_instructions=format_instructions)

response = chat.invoke(prompt)
print(response.content)

response_dict = output_parser.parse(text=response.content)
print(response_dict)

# LangChaint ConversationChain
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=chat, memory=memory)

res1 = conversation.predict(input="Hi, who are you? My name is Aleksandar!")
res2 = conversation.predict(input="Is it great day today?")
res3 = conversation.predict(input="Do you remember my name?")

print(conversation.memory.buffer)


