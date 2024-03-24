import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings

# OpenAI Library Example
client = OpenAI(api_key=openai_api_key)

messages = [
    {"role": "user", "content": "Hallo, who are you?"}
]

# result = client.chat.completions.create(messages=messages, model="gpt-3.5-turbo")
#
# print(result.choices[0].message.content)

# LangChain Model String Message Example

chat = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0.6)

# result = chat.invoke(input="Hallo, who are you?")
# print(result.content)

# LangChain Model using prompts

prompt_template_string = """
From given text separated by triple backticks extract following data:
part_of_the_day: Conclude what part of the day text is talking about.
weather: Make conclusion about what whether is described in the text.
Given text is: {text}
{format_instructions}
"""

format_instructions = "Give the result in json format."

text = """
We were walking through the shadow of death.
"""

prompt_template = ChatPromptTemplate.from_template(template=prompt_template_string)
prompt = prompt_template.format_prompt(text=text, format_instructions=format_instructions)

# result = chat.invoke(prompt)
# print(result.content)

# LangChain Model using output parser

part_of_the_day_schema = ResponseSchema(
    name="part_of_the_day",
    description="Conclude what part of the day text is talking about.")
weather_schema = ResponseSchema(
    name="weather",
    description="Make conclusion about what whether is described in the text.")

output_parser = StructuredOutputParser.from_response_schemas([part_of_the_day_schema, weather_schema])
format_instructions = output_parser.get_format_instructions()

prompt_template = ChatPromptTemplate.from_template(template=prompt_template_string)
prompt = prompt_template.format_prompt(text=text, format_instructions=format_instructions)

# result = chat.invoke(prompt)
# print(result.content)

# result_dict = output_parser.parse(result.content)
# print(result_dict)

# Conversation chain
memory = ConversationBufferMemory()
conversation_chain = ConversationChain(
    llm=chat,
    memory=memory,
    verbose=True
)

result1 = conversation_chain.predict(input="Hi my name is Aleksandar!")
result2 = conversation_chain.predict(input="It is nice weather today")
result3 = conversation_chain.predict(input="Do you remember my name?")

# print(conversation_chain.memory.buffer)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

embeddings_list = embeddings.embed_documents(texts=["What is your problem"])

pass
