
from openai import OpenAI

client = OpenAI(api_key='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')

result = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!!!"}],
    model="gpt-3.5-turbo")

print(result.choices[0].message.content)

