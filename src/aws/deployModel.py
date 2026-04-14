import os
from dotenv import load_dotenv
import sagemaker
from sagemaker.pytorch import PyTorch, PyTorchModel
from sagemaker.serializers import IdentitySerializer, JSONSerializer
from sagemaker.deserializers import JSONDeserializer

load_dotenv()

role = os.getenv("SAGEMAKER_ROLE_ARN")
source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cv"))

session = sagemaker.Session()

# ---- FIXED ESTIMATOR ----
estimator = PyTorch(
    entry_point="train.py",          
    source_dir=source_dir,               
    role=role,
    framework_version="2.1",
    py_version="py310",
    instance_count=1,
    instance_type="ml.m5.large",
    hyperparameters={
        "epochs": 15,
        "lr": 0.1
    },
    output_path="s3://driver-photo-bucket-554448410167-us-east-1-an/models"
)

# ---- TRAINING FROM S3 ----
estimator.fit({
    "training": "s3://driver-photo-bucket-554448410167-us-east-1-an/dataset"
})

print(estimator.model_data)

pytorch_model = PyTorchModel(
    model_data=estimator.model_data,   # IMPORTANT FIX
    role=role,
    framework_version="2.1",
    py_version="py310",
    entry_point="inference.py",
    source_dir=source_dir,
    sagemaker_session=session
)

predictor = pytorch_model.deploy(
    initial_instance_count=1,
    #instance_type="ml.g4dn.xlarge",
    instance_type="ml.m5.large",
    serializer=IdentitySerializer(content_type="application/x-image"),
    deserializer=JSONDeserializer()
)
