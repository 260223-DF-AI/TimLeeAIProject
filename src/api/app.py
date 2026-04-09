import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from pathlib import Path
import os

from src.llm.chatGPTAPI import get_report


frames = """
Frame 1:
safe driving (0.85)
radio usage (0.10)
talking to passenger (0.05)

Frame 2:
safe driving (0.82)
hair/makeup (0.10)
radio usage (0.08)

Frame 3:
safe driving (0.88)
talking to passenger (0.07)
radio usage (0.05)

Frame 4:
safe driving (0.79)
safe driving (0.79)
phone usage (0.21)

Frame 5:
phone usage (0.75)
safe driving (0.15)
talking to passenger (0.10)

Frame 6:
safe driving (0.81)
radio usage (0.12)
hair/makeup (0.07)

Frame 7:
safe driving (0.84)
talking to passenger (0.10)
radio usage (0.06)

Frame 8:
reaching behind (0.70)
safe driving (0.20)
radio usage (0.10)

Frame 9:
safe driving (0.83)
hair/makeup (0.10)
radio usage (0.07)

Frame 10:
drinking (0.77)
safe driving (0.14)
phone usage (0.09)
"""

app = FastAPI()

@app.post("/")
def post_root(file: UploadFile = File(...), text: str = Form()):

    # will model need to be created here, or will it already exist on sagemaker??
    #model = Model()

    # almost certainly this syntax will be different later (maybe more than 1 response for top 3 predictions?)
    #prediction, confidence = model(file, text)

    # buildPrompt will be in a different file
    #prompt = buildPrompt(prediction, confidence, text)
    #response = get_report(frames)
    # send the prompt that was generated to the LLM
    #response = LLM(prompt)

    # return the response, might need to reformat how the response is returned
    return "placeholder"