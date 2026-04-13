import os
from dotenv import load_dotenv
import sagemaker
from sagemaker.pytorch import PyTorch, PyTorchModel

# do these need to be in each funciton?
load_dotenv()
role = os.getenv("SAGEMAKER_ROLE_ARN")
session = sagemaker.Session()

# change this path if needed
source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cv"))


# def run_training(entry, epochs):
#     """
#     Create a model from scratch and train it on SageMaker. Stores the created model in S3 bucket.
#     entry: path to training data
#     epochs: number of training epochs
#     """
    
#     estimator = PyTorch(
#         entry_point=entry,
#         source_dir=source_dir,
#         role=role,
#         framework_version="2.1",
#         py_version="py310",
#         instance_count=1,
#         instance_type="ml.m5.large",

#         hyperparameters={
#             "epochs": epochs
#         },
#         output_path="s3://driver-photo-bucket-554448410167-us-east-1-an/models"
#     )

#     estimator.fit({
#         "training": "s3://driver-photo-bucket-554448410167-us-east-1-an/dataset"
#     })

#     model = PyTorchModel(
#         model_data=estimator.model_data,
#         role=role,
#         entry_point="inference.py",
#         source_dir=source_dir,
#         framework_version="2.1",
#         py_version="py310"
#     )

#     predictor = model.deploy(
#         initial_instance_count=1,
#         instance_type="ml.m5.large"
#     )
#     with open("cat1.jpg", "rb") as f:
#         payload = f.read()

#     response = predictor.predict(
#         payload,
#         initial_args={"ContentType": "application/x-image"}
#     )

#     print(response)
#     predictor.delete_endpoint()

def run_training(entry, epochs):
    estimator = PyTorch(
        entry_point=entry,
        source_dir=source_dir,
        role=role,
        framework_version="2.1",
        py_version="py310",
        instance_count=1,
        instance_type="ml.m5.large",
        hyperparameters={"epochs": epochs},
        output_path="s3://driver-photo-bucket-554448410167-us-east-1-an/models"
    )

    estimator.fit({
        "training": "s3://driver-photo-bucket-554448410167-us-east-1-an/dataset"
    })

    model = PyTorchModel(
        model_data=estimator.model_data,
        role=role,
        entry_point="inference.py",
        source_dir=source_dir,
        framework_version="2.1",
        py_version="py310"
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large"
    )

    try:
        image_path = os.path.join(os.path.dirname(__file__), "cat1.jpg")

        with open(image_path, "rb") as f:
            payload = f.read()

        response = predictor.predict(
            payload,
            initial_args={"ContentType": "application/x-image"}
        )

        print(response)

    finally:
        # ✅ always clean up
        predictor.delete_endpoint()


run_training("train.py", 5)
