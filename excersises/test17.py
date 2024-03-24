import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers import ResponseSchema
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Conversation Model
summarise_prompt_template_string = """
Summarise text delimited by triple backticks, and remove unnecessary details.
text is ```{text}```
{format_instructions}
"""

infer_prompt_template_string = """
From text delimited by triple backticks extract following data:
part_of_the_day: Part of the day that is mentioned in the text.
weather: Weather that is described in the given text.
text is ```{text}```
{format_instructions}
"""

transform_prompt_template_string = """
Translate from english to serbian text delimited by triple backticks:
text is ```{text}```
{format_instructions}
"""

expand_prompt_template_string = """
Expand the text delimited by triple backticks with new made up details, and give it in the form of essay.
text is ```{text}```
{format_instructions}
"""

format_instructions = "Give the result in json format."

text = """
We are walking on the shiny send while we were sweating because of the extreme heat outside. 
"""

summarise_prompt_string = summarise_prompt_template_string.format(text=text, format_instructions=format_instructions)
infer_prompt_string = infer_prompt_template_string.format(text=text, format_instructions=format_instructions)
transform_prompt_string = transform_prompt_template_string.format(text=text, format_instructions=format_instructions)
expand_prompt_string = expand_prompt_template_string.format(text=text, format_instructions=format_instructions)


# OpenAI API Library
client = OpenAI(api_key=openai_api_key)

summarise_result = client.chat.completions.create(
    messages=[{"role": "user", "content": summarise_prompt_string}],
    model="gpt-3.5-turbo")
infer_result = client.chat.completions.create(
    messages=[{"role": "user", "content": infer_prompt_string}],
    model="gpt-3.5-turbo")
transform_result = client.chat.completions.create(
    messages=[{"role": "user", "content": transform_prompt_string}],
    model="gpt-3.5-turbo")
expand_result = client.chat.completions.create(
    messages=[{"role": "user", "content": expand_prompt_string}],
    model="gpt-3.5-turbo")


print(summarise_result.choices[0].message.content)
print(infer_result.choices[0].message.content)
print(transform_result.choices[0].message.content)
print(expand_result.choices[0].message.content)

# LangChain Message Example
chat = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")

summarise_result_me = chat.invoke(input=summarise_prompt_string)
infer_result_me = chat.invoke(input=infer_prompt_string)
transform_result_me = chat.invoke(input=transform_prompt_string)
expand_result_me = chat.invoke(input=expand_prompt_string)

print(summarise_result_me.content)
print(infer_result_me.content)
print(transform_result_me.content)
print(expand_result_me.content)

# LangChain Prompts Example
summarise_prompt_template = ChatPromptTemplate.from_template(template=summarise_prompt_template_string)
infer_prompt_template = ChatPromptTemplate.from_template(template=infer_prompt_template_string)
transform_prompt_template = ChatPromptTemplate.from_template(template=transform_prompt_template_string)
expand_prompt_template = ChatPromptTemplate.from_template(template=expand_prompt_template_string)

prompt_template = ChatPromptTemplate.from_messages(
    messages=[
        summarise_prompt_template,
        infer_prompt_template,
        transform_prompt_template,
        expand_prompt_template
    ]
)

summarise_prompt_pe = summarise_prompt_template.format_prompt(text=text, format_instructions=format_instructions)
infer_prompt_pe = infer_prompt_template.format_prompt(text=text, format_instructions=format_instructions)
transform_prompt_pe = transform_prompt_template.format_prompt(text=text, format_instructions=format_instructions)
expand_prompt_pe = expand_prompt_template.format_prompt(text=text, format_instructions=format_instructions)

prompt_pe = prompt_template.format_prompt(text=text, format_instructions=format_instructions)

summarise_result_pe = chat.invoke(summarise_prompt_pe)
infer_result_pe = chat.invoke(infer_prompt_pe)
transform_result_pe = chat.invoke(transform_prompt_pe)
expand_result_pe = chat.invoke(expand_prompt_pe)
result_pe = chat.invoke(prompt_pe)

print(summarise_result_pe.content)
print(infer_result_pe.content)
print(transform_result_pe.content)
print(expand_result_pe.content)
print(result_pe.content)

# LangChain Output Parser Example
part_of_the_day_schema = ResponseSchema(
    name="part_of_the_day",
    description="Part of the day that is mentioned in the text."
)
weather_schema = ResponseSchema(
    name="weather",
    description="Weather that is described in the given text."
)

output_parser = StructuredOutputParser.from_response_schemas(response_schemas=[part_of_the_day_schema, weather_schema])
format_instructions_op = output_parser.get_format_instructions()

summarise_prompt_template_op = ChatPromptTemplate.from_template(template=summarise_prompt_template_string)
infer_prompt_template_op = ChatPromptTemplate.from_template(template=infer_prompt_template_string)
transform_prompt_template_op = ChatPromptTemplate.from_template(template=transform_prompt_template_string)
expand_prompt_template_op = ChatPromptTemplate.from_template(template=expand_prompt_template_string)

summarise_prompt_op = summarise_prompt_template_op.format_prompt(text=text, format_instructions=format_instructions)
infer_prompt_op = infer_prompt_template_op.format_prompt(text=text, format_instructions=format_instructions_op)
transform_prompt_op = transform_prompt_template_op.format_prompt(text=text, format_instructions=format_instructions)
expand_prompt_op = expand_prompt_template_op.format_prompt(text=text, format_instructions=format_instructions)

infer_result_op = chat.invoke(infer_prompt_op)
print(infer_result_op.content)

infer_result_dict_op = output_parser.parse(infer_result_op.content)
print(infer_result_dict_op)

# LangChain ConversationChain
memory = ConversationBufferMemory()
conversation_chain = ConversationChain(
    llm=chat,
    memory=memory
)

res1 = conversation_chain.predict(input="Hi, my name is Aleksandar!")
res2 = conversation_chain.predict(input="The weather outside is great, isn't it?")
res3 = conversation_chain.predict(input="Do you remember what was my name?")

print(conversation_chain.memory.buffer)




