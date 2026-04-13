# import os
# import torch
# from torch.utils.data import DataLoader
# import torch
# import torch.nn as nn
# from torchvision import datasets, transforms
# import shutil
# import sagemaker
# import tarfile

# TAR_NAME = 'model.tar.gz'
# LOCAL_MODEL_DIR = 'local_model'

# class CatOrDogModel(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

#         self.pool = nn.MaxPool2d(2, 2)

#         self.fc1 = nn.Linear(64 * 28 * 28, 128)
#         self.fc2 = nn.Linear(128, 2)

#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))
#         x = self.pool(torch.relu(self.conv3(x)))

#         x = x.view(x.size(0), -1)

#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)

#         return x


# def train():
#     print("starting training")
#     training_dir = os.environ["SM_CHANNEL_TRAINING"]
#     model_dir = os.environ["SM_MODEL_DIR"]

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     transform = transforms.Compose([
#         transforms.Resize((224,224)),
#         transforms.ToTensor()
#     ])

#     dataset = datasets.ImageFolder(
#         training_dir,
#         transform=transform
#     )

#     print("dataset size:", len(dataset))
#     print("classes:", dataset.classes)

#     loader = DataLoader(dataset, batch_size=32, shuffle=True)

#     model = CatOrDogModel().to(device)
#     model.train()

#     loss_fn = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters())
#     print("first loop")
#     for epoch in range(5):
        
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

#     os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
#     model_path = os.path.join(LOCAL_MODEL_DIR, 'model.pth')

#     torch.save(model.state_dict(), model_path)

#     code_dir = os.path.join(LOCAL_MODEL_DIR, 'code')
#     os.makedirs(code_dir, exist_ok=True)

#     if os.path.exists('cv/inference.py'):
#         shutil.copy('cv/inference.py', os.path.join(code_dir, 'inference.py'))

#     with tarfile.open(TAR_NAME, "w:gz") as tar:
#         tar.add(model_path, arcname='model.pth')
#         tar.add(code_dir, arcname='code')

#     print(f"Saved model to {TAR_NAME}")

#     try:
#         session = sagemaker.Session()
#         bucket = session.default_bucket()
#         print(f"Bucket: {bucket}")
#     except Exception as e:
#         print(e)
#         exit(1)

#     s3_prefix = 'CatDogTest'
#     s3_model_path = session.upload_data(path=TAR_NAME, bucket=bucket, key_prefix=s3_prefix)

#     print(f"Uploaded model to {s3_model_path}")
    
#     # model_path = os.path.join(model_dir, "model.pth")
#     # torch.save(model.state_dict(), model_path)


# train()
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
class CatOrDogModel(nn.Module):
    def __init__(self):
        super().__init__()

        # ---- Feature extractor ----
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # 🔥 replaces huge flatten (KEY FIX)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ---- Classifier (tiny now) ----
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 224 → 112
        x = self.pool(F.relu(self.conv2(x)))   # 112 → 56
        x = self.pool(F.relu(self.conv3(x)))   # 56 → 28

        # 🔥 compress spatial dimensions safely
        x = self.global_pool(x)                # 28x28 → 1x1

        x = torch.flatten(x, 1)                # (B, 64)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    training_dir = os.environ["SM_CHANNEL_TRAINING"]
    model_dir = os.environ["SM_MODEL_DIR"]

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(training_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CatOrDogModel().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model.train()

    for epoch in range(10):
        total_loss = 0.0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"epoch {epoch} loss: {total_loss:.4f}")

    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))

if __name__ == "__main__":
    main()
