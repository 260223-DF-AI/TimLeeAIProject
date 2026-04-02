import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

print()
print("--- Device Configuration ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print()
print("--- Transformations ---")
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,)) 
])

print()
print("--- Downloading/Loading FashionMNIST Dataset ---")
train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

print()
print("--- Define Model ---")
class FashionModel(nn.Module):
    def __init__(self):
        super(FashionModel, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear( 28 * 28, 128),
            nn.ReLU(),
            # nn.Dropout(p=0.2), # dropout is turning off 20% of the nodes each epoch
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

print()
print(" --- Instantiating Model ---")
model = FashionModel().to(device)
print(model)

print()
print(" --- Optimizer and Loss Function ---")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train_one_epoch(model, dataloader, optimzer, criterion):
    print()
    print("--- Training One Epoch ---")
    
    model.train() # The model is ready to learn (and dropout is active)
    for batch, (x,y) in enumerate(dataloader):
        
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = criterion(pred, y)
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()

        if batch % 100 == 0:
            print(f"Batch {batch} - Loss: {loss}")
            if batch >= 700: break

    #print(batch)

def evaluate(model, dataloader, criterion):
    print()
    print("--- Evaluating Model ---")
    
    model.eval() # The model is not learning, and dropout is inactive

    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            test_loss += criterion(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {correct / len(dataloader.dataset):.4f}")

if __name__ == "__main__":
    # print()
    # train_one_epoch(model, train_loader, optimizer, criterion)
    # evaluate(model, test_loader, criterion)