# from main_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage

chat = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="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")

print(chat.invoke("What is your name?"))

llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key="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")

print(llm.invoke("What is your name again?"))

response = chat.invoke(input=[
    SystemMessage("You are an historian, and answer everything from that perspective."),
    HumanMessage("What time is it?")
])

print(response)

