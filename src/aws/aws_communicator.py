import os
from dotenv import load_dotenv
import sagemaker
from sagemaker.pytorch import PyTorch, PyTorchModel
from sagemaker.serializers import IdentitySerializer
from sagemaker.deserializers import JSONDeserializer
import torch

load_dotenv()
role = os.getenv("SAGEMAKER_ROLE_ARN")
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
        instance_type="ml.m5.large",
        hyperparameters={
            "epochs": 200, # this is the max number, it should stop well before this
            "lr": 0.001, # initial LR only
            "patience": 20, # early stopping: epochs without improvement
            "lr_patience": 5,  # scheduler: epochs before reducing LR
            "lr_factor": 0.5, # scheduler: multiply LR by this on plateau
        },
        output_path="s3://driver-photo-bucket-554448410167-us-east-1-an/models"
    )

    estimator.fit({
        "training": "s3://driver-photo-bucket-554448410167-us-east-1-an/imgs/train",
        "validation": "s3://driver-photo-bucket-554448410167-us-east-1-an/imgs/val",
    })
    return estimator.model_data


def deploy_model(model_data):
    pytorch_model = PyTorchModel(
        model_data=model_data,
        role=role,
        framework_version="2.1",
        py_version="py310",
        entry_point="inference.py",
        source_dir=source_dir,
        sagemaker_session=session
    )

    predictor = pytorch_model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
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
    logits = torch.tensor(output)
    probs = torch.softmax(logits, dim=1)
    confidence, pred_class = torch.max(probs, dim=1)

    return {
        "class": int(pred_class.item()),
        "confidence": float(confidence.item()),
        "probabilities": probs.tolist()
    }