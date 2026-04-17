import os
from dotenv import load_dotenv
import sagemaker
from sagemaker.pytorch import PyTorch, PyTorchModel
from sagemaker.serializers import IdentitySerializer
from sagemaker.deserializers import JSONDeserializer
import torch

load_dotenv()
role = os.getenv("SAGEMAKER_ROLE_ARN")
#source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cv"))
session = sagemaker.Session()


def train_model():
    estimator = PyTorch(
        entry_point="train.py",
        source_dir=source_dir,
        role=role,
        framework_version="2.1",
        py_version="py310",
        instance_count=1,
        instance_type="ml.g4dn.xlarge",
        hyperparameters={
            "epochs": 30, # this is the max number, it should stop well before this
            "lr": 0.001, # initial LR only
            "patience": 7, # early stopping: epochs without improvement
            "lr_patience": 5, # scheduler: epochs before reducing LR
            "lr_factor": 0.5, # scheduler: multiply LR by this on plateau
        },
        output_path="s3://driver-photo-bucket-554448410167-us-east-1-an/models"
    )

    estimator.fit({
        "training": "s3://driver-photo-bucket-554448410167-us-east-1-an/dataset/train",
        "validation": "s3://driver-photo-bucket-554448410167-us-east-1-an/dataset/val",
    },
    logs=True)
    return estimator.model_data


def deploy_model(model_data):
    pytorch_model = PyTorchModel(
        model_data=model_data,
        role=role,
        framework_version="2.1",
        py_version="py310",
        entry_point="cv/inference.py",
        source_dir=source_dir,
        sagemaker_session=session
    )

    predictor = pytorch_model.deploy(
        initial_instance_count=1,
        instance_type="ml.g4dn.xlarge",
        #instance_type="ml.m5.large",
        serializer=IdentitySerializer(content_type="application/x-image"),
        deserializer=JSONDeserializer()
    )
    return predictor.endpoint_name


def predict_model(payload):
    predictor = sagemaker.predictor.Predictor(
        endpoint_name=session.sagemaker_client.list_endpoints()["Endpoints"][0]["EndpointName"],
        sagemaker_session=session,
        serializer=IdentitySerializer(content_type="application/x-image"),
        deserializer=JSONDeserializer()
    )

    output = predictor.predict(payload)

    # Convert to tensor and compute probabilities
    return output

def delete_endpoint():
    endpoint_name = session.sagemaker_client.list_endpoints()["Endpoints"][0]["EndpointName"]

    # delete endpoint
    session.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)

    print(f"Deleted endpoint: {endpoint_name}")
    return "Deleted endpoint"