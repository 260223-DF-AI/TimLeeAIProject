from openai import OpenAI
from dotenv import load_dotenv
#import prompt
load_dotenv()

client = OpenAI()

with open("prompt.txt", "r", encoding="utf-8") as f:
    prompt = f.read()

prompt += """
c0, 95%
c0, 92%
c2, 90%
c0, 97%
c0, 86%
c6, 98%
c0, 84%
c0, 87%
c0, 94%
c5, 98%
"""

response = client.responses.create(
    model="gpt-5-mini",
    input=prompt
)

print(response.output_text)
