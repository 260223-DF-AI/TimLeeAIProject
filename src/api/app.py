import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import HTMLResponse
from pathlib import Path
import os
from src.llm.chatGPTAPI import get_report
from src.aws.aws_communicator import train_model, deploy_model, predict_model, delete_endpoint
from src.database import process_database, database_core
from typing import List
from datetime import datetime

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

app = FastAPI()

# @app.post("/analyze")
# def post_root(directory: str):
#     results = []

#     for filename in os.listdir(directory):
#         file_path = os.path.join(directory, filename)

#         if not os.path.isfile(file_path):
#             continue

#         with open(file_path, "rb") as f:
#             image_bytes = f.read()

#         results.append(predict_model(image_bytes))
#     return get_report(results)

@app.post("/analyze")
def post_root(directory: str):
    if not os.path.isdir(directory):
        return {"error": "Invalid directory"}

    results = []

    for filename in os.listdir(directory):

        file_path = os.path.join(directory, filename)

        if not os.path.isfile(file_path):
            continue

        with open(file_path, "rb") as f:
            image_bytes = f.read()

        prediction = predict_model(image_bytes)
        results.append(prediction)

    # Generate report
    print("generating report")
    report = get_report(results)

    # Ensure reports folder exists
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)

    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path = os.path.join(reports_dir, f"{timestamp}.txt")

    # Write report to file
    print("writing report")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(report)

    return {
        "message": "Report generated",
        "file": file_path
    }

@app.post("/train")
def post_train(background_tasks: BackgroundTasks):
    http_id = process_database.log_http_request("/train")
    background_tasks.add_task(train_model)
    return {"status": "training started"}

@app.post("/deploy")
def post_deploy(model: str = Form()):
    http_id = process_database.log_http_request("/deploy")
    return deploy_model(model)

@app.post("/predict")
def post_predict(file: UploadFile = File(...)):
    http_id = process_database.log_http_request("/predict")
    image_bytes = file.file.read()

    return predict_model(image_bytes)

@app.delete("/close_endpoint")
def delete_close_endpoint():
    #http_id = process_database.log_http_request("/closeEndpoint")
    return delete_endpoint()
@app.post("/bug_example")
def post_bug_example(files: List[UploadFile] = File(...)):
    return "glitch example"