import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate

chat = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0.6)

prompt_template_string = ("From the text delimited by backticks extract following data:"
                          "part_of_the_day: Make conclusion about part of the day text is talking about."
                          "weather: Make conclusion about weather that text is talking about."
                          "Given text is: {text}"
                          "{format_instructions}")
format_instructions = "Create response in json format."

text = ("We were talking outside of the house while sun was behind the hills in front of us. We couldn't"
        "here each other because of the sound trees and leafs produced because of their intensive movements.")

prompt_string = prompt_template_string.format(text=text, format_instructions=format_instructions)

# response = chat.invoke(prompt_string)
#
# print(response.content)

# Prompt object
prompt_template = ChatPromptTemplate.from_template(prompt_template_string)

prompt = prompt_template.format_prompt(text=text, format_instructions=format_instructions)

chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are helpful assistant by the name Oskar"),
        ("human", "Hi, my name is Aleksandar, who are you?"),
        ("ai", "I am Oskar"),
        ("human", "What was my name?")
    ]
)

prompt = chat_prompt_template.format_prompt()

response = chat.invoke(prompt)

print(response.content)



pass
