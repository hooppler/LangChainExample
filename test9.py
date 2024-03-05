import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

chat = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0.6)

prompt_template_string = ("From text delimited by triple backticks, extract following data:"
                          "part_of_the_day: What part of the day text is talking about?"
                          "weather: What weather conditions text describes."
                          "Given text is: ```{text}```"
                          "{format_instructions}")
format_instructions = "Format result in the json format."

text = "We were walking in the dark while we were hardly able to se because of water falling into our eyes"

# Prompt string
prompt_string = prompt_template_string.format(text=text, format_instructions=format_instructions)

result = chat.invoke(prompt_string)

print(result.content)

# Prompt object
prompt_template = ChatPromptTemplate.from_template(template=prompt_template_string)
prompt = prompt_template.format_prompt(text=text, format_instructions=format_instructions)

result1 = chat.invoke(prompt)

print(result1.content)

# Output parser
part_of_the_day_schema = ResponseSchema(name="part_of_the_day", description="What part of the day text is talking about?")
weather_schema = ResponseSchema(name="weather", description="What weather conditions text describes.")

output_parser = StructuredOutputParser.from_response_schemas([part_of_the_day_schema, weather_schema])
format_instructions = output_parser.get_format_instructions()

prompt = prompt_template.format_prompt(text=text, format_instructions=format_instructions)

result2 = chat.invoke(prompt)

print(result2.content)

result_dict = output_parser.parse(result2.content)

print(result_dict)

# Conversation chain
memory = ConversationBufferMemory()

conversation = ConversationChain(llm=chat, memory=memory, verbose=True)

res1 = conversation.predict(input="Hi, my name is Aleksandar, what is yours?")
res2 = conversation.predict(input="What is 3+5 ?")
res3 = conversation.predict(input="Do you remember my name?")

print(conversation.memory)











