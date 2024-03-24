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

# Messages Model
infer_prompt_template_string = """
From the text delimited by triple backticks, extract following items:
part_of_the_day: Part of the day described in the given text.
weather: Weather that is described in the text.
Given text is: ```{text}```
{format_instructions}
"""

text = """
We were walking on the bright, shinny, hot sand. Drops of sweat was falling to the ground because of intense heat.
"""

# OpenAI Model Call
client = OpenAI(api_key=openai_api_key)

infer_prompt_string = infer_prompt_template_string.format(text=text, format_instructions="")

# result_mc = client.chat.completions.create(
#     messages=[{"role": "user", "content": infer_prompt_string}],
#     model="gpt-3.5-turbo")
# print(result_mc.choices[0].message.content)

# LangChain Single Message
chat = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")

# result_sm = chat.invoke(input=infer_prompt_string)
# print(result_sm.content)

# LangChain Prompt Object
infer_prompt_template = ChatPromptTemplate.from_template(template=infer_prompt_template_string)
infer_prompt_po = infer_prompt_template.format_prompt(text=text, format_instructions="")

# result_po = chat.invoke(infer_prompt_po)
# print(result_po.content)

# LangChain Output Parser
part_of_the_day_schema = ResponseSchema(
    name="part_of_the_day",
    description="Part of the day described in the given text.")
weather_schema = ResponseSchema(
    name="weather",
    description="Weather that is described in the text.")

output_parser = StructuredOutputParser.from_response_schemas(response_schemas=[part_of_the_day_schema, weather_schema])
format_instructions = output_parser.get_format_instructions()

infer_prompt_op = infer_prompt_template.format_prompt(text=text, format_instructions=format_instructions)

response_op = chat.invoke(infer_prompt_op)
print(response_op)

response_dict_op = output_parser.parse(response_op.content)
print(response_dict_op)

# LangChain Conversation Chain
memory = ConversationBufferMemory()
conversation_chain = ConversationChain(
    llm=chat,
    memory=memory
)

res1 = conversation_chain.predict(input="Hi, my name is Aleksandar")
res2 = conversation_chain.predict(input="Today is nice day, what do you think?")
res3 = conversation_chain.predict(input="Do you remember my name?")

print(conversation_chain.memory.buffer)


