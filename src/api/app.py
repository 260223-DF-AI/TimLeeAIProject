import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
# from pathlib import Path
# import os
# 


app = FastAPI()

@app.post("/")
def post_root(file: UploadFile = File(...), text: str = Form()):

    # will model need to be created here, or will it already exist on sagemaker??
    #model = Model()

    # almost certainly this syntax will be different later (maybe more than 1 response for top 3 predictions?)
    #prediction, confidence = model(file, text)

    # buildPrompt will be in a different file
    #prompt = buildPrompt(prediction, confidence, text)

    # send the prompt that was generated to the LLM
    #response = LLM(prompt)

    # return the response, might need to reformat how the response is returned
    return "response"