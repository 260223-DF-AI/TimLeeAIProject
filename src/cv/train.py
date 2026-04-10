import os
import torch
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchvision import datasets, transforms

#from src.cv.model import CatDogModel

class CatOrDogModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def train():
    print("starting training")
    training_dir = os.environ["SM_CHANNEL_TRAINING"]
    model_dir = os.environ["SM_MODEL_DIR"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(
        training_dir,
        transform=transform
    )

    print("dataset size:", len(dataset))
    print("classes:", dataset.classes)

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = CatOrDogModel().to(device)
    model.train()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    print("first loop")
    for epoch in range(5):
        
        total_loss = 0.0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"epoch {epoch} loss: {total_loss:.4f}", flush=True)

    # torch.save(model.state_dict(), f"{model_dir}/model.pth")
    # print("model saved")

train()
