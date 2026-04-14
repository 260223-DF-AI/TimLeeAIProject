import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torch.nn as nn
from model import CatOrDogModel
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=0.001)

args = parser.parse_args()

epochs = args.epochs
lr = args.lr


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
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
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
