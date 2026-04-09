from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()


def get_report(frames):
    with open("src/llm/prompt2.txt", "r", encoding="utf-8") as f:
        prompt = f.read()
    prompt += frames
    response = client.responses.create(
        model="gpt-5-nano",
        input=prompt)
    return response

# prompt += """
# Frame 1:
# safe driving (0.78)
# talking to passenger (0.14)
# radio usage (0.08)

# Frame 2:
# safe driving (0.81)
# radio usage (0.11)
# talking to passenger (0.08)

# Frame 3:
# phone usage (0.72)
# safe driving (0.18)
# radio usage (0.10)

# Frame 4:
# phone usage (0.84)
# safe driving (0.09)
# talking to passenger (0.07)

# Frame 5:
# phone usage (0.76)
# radio usage (0.13)
# safe driving (0.11)

# Frame 6:
# safe driving (0.69)
# talking to passenger (0.17)
# radio usage (0.14)

# Frame 7:
# drinking (0.66)
# safe driving (0.21)
# reaching behind (0.13)

# Frame 8:
# phone usage (0.81)
# safe driving (0.12)
# radio usage (0.07)

# Frame 9:
# safe driving (0.74)
# talking to passenger (0.16)
# hair/makeup (0.10)

# Frame 10:
# phone usage (0.79)
# safe driving (0.13)
# radio usage (0.08)
# """

# prompt += """
# Frame 1:
# safe driving (0.85)
# radio usage (0.10)
# talking to passenger (0.05)

# Frame 2:
# safe driving (0.82)
# hair/makeup (0.10)
# radio usage (0.08)

# Frame 3:
# safe driving (0.88)
# talking to passenger (0.07)
# radio usage (0.05)

# Frame 4:
# safe driving (0.79)
# safe driving (0.79)
# phone usage (0.21)

# Frame 5:
# phone usage (0.75)
# safe driving (0.15)
# talking to passenger (0.10)

# Frame 6:
# safe driving (0.81)
# radio usage (0.12)
# hair/makeup (0.07)

# Frame 7:
# safe driving (0.84)
# talking to passenger (0.10)
# radio usage (0.06)

# Frame 8:
# reaching behind (0.70)
# safe driving (0.20)
# radio usage (0.10)

# Frame 9:
# safe driving (0.83)
# hair/makeup (0.10)
# radio usage (0.07)

# Frame 10:
# drinking (0.77)
# safe driving (0.14)
# phone usage (0.09)
# """

# response = client.responses.create(
#     model="gpt-5-nano",
#     input=prompt
# )

# print(response.output_text)
# print(response.usage.total_tokens)