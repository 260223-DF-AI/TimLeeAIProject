import pandas as pd
from fastapi import FastAPI, APIRouter, HTTPException
from pathlib import Path
import os

app = FastAPI()

@app.post("/")
def post_root():

    prediction, confidence = 

    return "placeholder"