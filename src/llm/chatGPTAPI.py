from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()


def get_report(frames):
    with open("src/llm/prompt3.txt", "r", encoding="utf-8") as f:
        prompt = f.read()
    for frame in frames:
        prompt += str(frame)
    response = client.responses.create(
        model="gpt-5-nano",
        input=prompt)
    
    return response.output[1].content[0].text

