import os
from dotenv import load_dotenv
import sagemaker
from sagemaker.pytorch import PyTorch, PyTorchModel

load_dotenv()

role = os.getenv("SAGEMAKER_ROLE_ARN")

session = sagemaker.Session()

source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cv"))

estimator = PyTorch(
    entry_point="train.py",
    source_dir=source_dir,
    role=role,
    framework_version="2.1",
    py_version="py310",
    instance_count=1,
    instance_type="ml.m5.large",

    hyperparameters={
        "epochs": 5
    }
)

estimator.fit({
    "training": "s3://driver-photo-bucket-554448410167-us-east-1-an/dataset"
})
