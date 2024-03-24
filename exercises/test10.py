import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers import ResponseSchema
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

chat = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0.5)

# String Input
prompt_template_string = ("From text delimited by triple backticks, extract following data:"
                          "part_of_the_day: What part of the day text is talking about?"
                          "weather: What weather text is describing?"
                          "Given text is: ```{text}```"
                          "{format_instruction}")
text = ("We were walking at the middle of the mountain and it was clearly visible the lake in front of us. We were "
        "sweating because of high temperature outside.")
format_instruction = "Give the results in json form."

prompt_string = prompt_template_string.format(text=text, format_instruction=format_instruction)

# result = chat.invoke(prompt_string)
#
# print(result.content)

# Prompt object

prompt_template = ChatPromptTemplate.from_template(template=prompt_template_string)
prompt = prompt_template.format_prompt(text=text, format_instruction=format_instruction)

# result = chat.invoke(prompt)
#
# print(result.content)

# Format objects
part_of_the_day_schema = ResponseSchema(name="part_of_the_day", description="What part of the day text is talking about?")
weather_schema = ResponseSchema(name="weather", description="What weather text is describing?")

output_parser = StructuredOutputParser.from_response_schemas([part_of_the_day_schema, weather_schema])
format_instruction = output_parser.get_format_instructions()

prompt = prompt_template.format_prompt(text=text, format_instruction=format_instruction)

result = chat.invoke(prompt)

# print(result.content)
#
# result_dict = output_parser.parse(result.content)
#
# print(result_dict)

# Conversational Chain
memory = ConversationBufferMemory()

conversation_chain = ConversationChain(llm=chat, memory=memory, verbose=True)

conversation_chain.predict(input="Hi there, my name is Aleksandar.")
conversation_chain.predict(input="What is surname of scientist Einstein?")
conversation_chain.predict(input="Do you remember my name ?")

print(conversation_chain.memory.buffer)







