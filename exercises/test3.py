import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from openai import OpenAI

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

result = client.chat.completions.create(messages=[{"role": "user", "content": "Hi there who are you?"}], model="gpt-3.5-turbo")
print(type(result))
print(result.choices[0].message.content)

res = client.embeddings.create(input="This is some example text.", model="text-embedding-3-large")

print(res.data[0].embedding)

image = client.images.generate(prompt="House in the forrest", model="dall-e-2")

print(image.data[0])

speech = client.audio.speech.create(input="Hi, how are you today?", model="tts-1", voice="alloy")

client.with_streaming_response.audio("MyExampleVoice")

print(speech.content)

