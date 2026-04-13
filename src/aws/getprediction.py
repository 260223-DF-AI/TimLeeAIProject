import boto3
import sagemaker
from sagemaker.serializers import IdentitySerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.pytorch import PyTorchPredictor
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["AWS_DEFAULT_REGION"] = os.getenv("AWS_DEFAULT_REGION")

endpoint_name = "pytorch-inference-2026-04-13-19-38-08-513"

predictor = sagemaker.predictor.Predictor(
    endpoint_name=endpoint_name,
    sagemaker_session=sagemaker.Session(),
    serializer=IdentitySerializer(content_type="application/x-image"),
    deserializer=JSONDeserializer()
)

image_path = "src/aws/cat1.jpg"

with open(image_path, "rb") as f:
    payload = f.read()

response = predictor.predict(payload)

print("Response:", response)

image_path = "src/aws/dog1.jpg"

with open(image_path, "rb") as f:
    payload = f.read()

response = predictor.predict(payload)

print("Response:", response)
