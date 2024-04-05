import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Instruction examples
summarize_prompt_template_string = """
Summarize text delimited by triple backticks, in at the most 25 words focusing on the most important details.
Given text is: ```{text}```
{format_instructions}
"""

infer_prompt_template_string = """
Form provided text delimited by triple backticks extract following items:
part_of_the_day: Part of the day that is described in the text.
weather: Weather that is described in the text.
Given text is: ```{text}```
{format_instructions}
"""

transform_prompt_template_string = """
Transform text delimited by triple backticks to an scientific form.
Given text is: ```{text}```
{format_instructions}
"""

expand_prompt_template_string = """
Create larger story from the text delimited by triple backtick, adding more made up details.
Given text is: ```{text}```
{format_instructions}
"""

format_instructions = "Create response in json format."

text = """
We were walking through hot shinny sand looking to the sky, while we were sweating because of heat outside. 
"""

# OpenAI Responses
client = OpenAI(api_key=openai_api_key)

summarize_prompt_string = summarize_prompt_template_string.format(text=text, format_instructions="")
infer_prompt_string = infer_prompt_template_string.format(text=text, format_instructions=format_instructions)
transform_prompt_string = transform_prompt_template_string.format(text=text, format_instructions="")
expand_prompt_string = expand_prompt_template_string.format(text=text, format_instructions="")

summarize_response_or = client.chat.completions.create(
    messages=[{"role": "user", "content": summarize_prompt_string}],
    model="gpt-3.5-turbo"
)
infer_response_or = client.chat.completions.create(
    messages=[{"role": "user", "content": infer_prompt_string}],
    model="gpt-3.5-turbo"
)
transform_response_or = client.chat.completions.create(
    messages=[{"role": "user", "content": transform_prompt_string}],
    model="gpt-3.5-turbo"
)
expand_response_or = client.chat.completions.create(
    messages=[{"role": "user", "content": expand_prompt_string}],
    model="gpt-3.5-turbo"
)

print(summarize_response_or.choices[0].message.content)
print(infer_response_or.choices[0].message.content)
print(transform_response_or.choices[0].message.content)
print(expand_response_or.choices[0].message.content)

# LangChain Message Example
chat = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")

summarize_response_me = chat.invoke(input=summarize_prompt_string)
infer_response_me = chat.invoke(input=infer_prompt_string)
transform_response_me = chat.invoke(input=transform_prompt_string)
expand_response_me = chat.invoke(input=expand_prompt_string)

print(summarize_response_me.content)
print(infer_response_me.content)
print(transform_response_me.content)
print(expand_response_me.content)

# LangChain Prompt Example

summarize_prompt_template = ChatPromptTemplate.from_template(template=summarize_prompt_template_string)
infer_prompt_template = ChatPromptTemplate.from_template(template=infer_prompt_template_string)
transform_prompt_template = ChatPromptTemplate.from_template(template=transform_prompt_template_string)
expand_prompt_template = ChatPromptTemplate.from_template(template=expand_prompt_template_string)

summarize_prompt = summarize_prompt_template.format_prompt(text=text, format_instructions="")
infer_prompt = infer_prompt_template.format_prompt(text=text, format_instructions=format_instructions)
transform_prompt = transform_prompt_template.format_prompt(text=text, format_instructions="")
expand_prompt = expand_prompt_template.format_prompt(text=text, format_instructions="")

summarize_response_pe = chat.invoke(input=summarize_prompt)
infer_response_pe = chat.invoke(input=infer_prompt)
transform_response_pe = chat.invoke(input=transform_prompt)
expand_response_pe = chat.invoke(input=expand_prompt)

print(summarize_response_pe.content)
print(infer_response_pe.content)
print(transform_response_pe.content)
print(expand_response_pe.content)

# LangChat Output Parser Example
part_of_the_day_schema = ResponseSchema(
    name="part_of_the_day",
    description="Part of the day that is described in the text."
)
weather_schema = ResponseSchema(
    name="weather",
    description="Weather that is described in the text."
)

output_parser = StructuredOutputParser.from_response_schemas(response_schemas=[part_of_the_day_schema, weather_schema])
format_instructions = output_parser.get_format_instructions()

infer_prompt_op = infer_prompt_template.format_prompt(text=text, format_instructions=format_instructions)

infer_result_op = chat.invoke(input=infer_prompt_op)
print(infer_result_op.content)

infer_result_dict_op = output_parser.parse(text=infer_result_op.content)
print(infer_result_dict_op)

# LangChain Conversation Chain Example
memory = ConversationBufferMemory()
conversation_chain = ConversationChain(
    llm=chat,
    memory=memory
)

res1 = conversation_chain.predict(input="Hi, my name is Aleksandar")
res2 = conversation_chain.predict(input="The weather is bed isn't it?")
res3 = conversation_chain.predict(input="Du you remember my name?")

print(conversation_chain.memory.buffer)



