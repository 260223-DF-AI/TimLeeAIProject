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
    entry_point="train.py",          # MUST exist
    source_dir=source_dir,                # folder containing train.py + model code
    role=role,
    framework_version="2.1",
    py_version="py310",
    instance_count=1,
    instance_type="ml.m5.large",
    hyperparameters={
        "epochs": 10,
        "lr": 0.01
    },
    output_path=f"s3://{session.default_bucket()}/models"
)

# ---- TRAINING FROM S3 ----
estimator.fit({
    "training": "s3://driver-photo-bucket-554448410167-us-east-1-an/dataset"
})

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

# image_path = os.path.join(os.path.dirname(__file__), "cat1.jpg")

# with open(image_path, "rb") as f:
#     payload = f.read()

# response = predictor.predict(
#     payload
# )
# print(response)
# predictor.delete_endpoint()

# import os
# import shutil
# import sagemaker
# import tarfile
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# from sagemaker.pytorch import PyTorch, PyTorchModel
# from sagemaker.serializers import JSONSerializer
# from sagemaker.deserializers import JSONDeserializer
# from src.cv.model import CatOrDogModel
# from dotenv import load_dotenv

# # do these need to be in each funciton?
# load_dotenv()
# role = os.getenv("SAGEMAKER_ROLE_ARN")

# USE_GPU = False

# TRAIN_DEVICE = 'ml.g4dn.xlarge' if USE_GPU else 'ml.m5.large'
# DEPLOY_DEVICE = 'ml.m5.large'

# LOCAL_MODEL_DIR = 'local_model'
# TAR_NAME = 'model.tar.gz'

# EPOCHS = 10
# LEARNING_RATE = 0.01

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"running on {device}")

# session = sagemaker.Session()

# estimator = PyTorch(
#         entry_point=entry,
#         source_dir=source_dir,
#         role=role,
#         framework_version="2.1",
#         py_version="py310",
#         instance_count=1,
#         instance_type="ml.m5.large",
#         hyperparameters={"epochs": epochs},
#         output_path="s3://driver-photo-bucket-554448410167-us-east-1-an/models"
#     )

# estimator.fit({
#         "training": "s3://driver-photo-bucket-554448410167-us-east-1-an/dataset"
#     })

# training_dir = os.environ["SM_CHANNEL_TRAINING"]
# model_dir = os.environ["SM_MODEL_DIR"]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transform = transforms.Compose([
#         transforms.Resize((224,224)),
#         transforms.ToTensor()
#     ])

# dataset = datasets.ImageFolder(
#         training_dir,
#         transform=transform
#     )

# print("dataset size:", len(dataset))
# print("classes:", dataset.classes)

# loader = DataLoader(dataset, batch_size=32, shuffle=True)

# model = CatOrDogModel().to(device)
# loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# print("Start the training loop...")

# model.train()
# for epoch in range(5):
        
#         total_loss = 0.0

#         for x, y in loader:
#             x, y = x.to(device), y.to(device)

#             pred = model(x)
#             loss = loss_fn(pred, y)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         print(f"epoch {epoch} loss: {total_loss:.4f}", flush=True)

# os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
# model_path = os.path.join(LOCAL_MODEL_DIR, 'model.pth')

# torch.save(model.state_dict(), model_path)

# code_dir = os.path.join(LOCAL_MODEL_DIR, 'code')
# os.makedirs(code_dir, exist_ok=True)

# if os.path.exists('../cv/inference.py'):
#     shutil.copy('../cv/inference.py', os.path.join(code_dir, 'inference.py'))

# with tarfile.open(TAR_NAME, "w:gz") as tar:
#     tar.add(model_path, arcname='model.pth')
#     tar.add(code_dir, arcname='code')

# print(f"Saved model to {TAR_NAME}")

# try:
#     session = sagemaker.Session()
#     bucket = session.default_bucket()
#     print(f"Bucket: {bucket}")
# except Exception as e:
#     print(e)
#     exit(1)

# s3_prefix = 'CatOrDog'
# s3_model_path = session.upload_data(path=TAR_NAME, bucket=bucket, key_prefix=s3_prefix)

# print(f"Uploaded model to {s3_model_path}")

# pytorch_model = PyTorchModel(
#     model_data=s3_model_path,
#     role=role,
#     framework_version='2.0.0',
#     py_version='py310',
#     entry_point='inference.py',
#     sagemaker_session=session
# )

# predictor = pytorch_model.deploy(
#     initial_instance_count=1,
#     instance_type=DEPLOY_DEVICE,
#     serializer=JSONSerializer(),
#     deserializer=JSONDeserializer()
# )

# predictor = model.deploy(
#         initial_instance_count=1,
#         instance_type="ml.m5.large"
#     )

# try:
#     image_path = os.path.join(os.path.dirname(__file__), "cat1.jpg")

#     with open(image_path, "rb") as f:
#         payload = f.read()

#     response = predictor.predict(
#         payload,
#         initial_args={"ContentType": "application/x-image"}
#     )

#     print(response)

# finally:
#         # ✅ always clean up
#     predictor.delete_endpoint()

# print(response)

# predictor.delete_endpoint()