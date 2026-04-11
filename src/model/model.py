"""Core code for training the CV model.

Data will need to have been processed/partitioned in database/process_database.py first."""

# main imports
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
from torch.utils.tensorboard import SummaryWriter

# project imports
from src.utils import logger
from src.paths import DATA_ROOT

TRAIN_DATA = DATA_ROOT / "imgs/train"
VAL_DATA = DATA_ROOT / "imgs/val"
TEST_DATA = DATA_ROOT / "imgs/test"
MODEL_PATH = DATA_ROOT / "model/DistractedDriverModel.pth"
LOG_DIR = DATA_ROOT / "model/logs"

logger = logger.setup_logger(__name__, "debug")

data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(45),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

try:
    train_dataset = datasets.ImageFolder(root=TRAIN_DATA, transform=data_transforms)
    val_dataset = datasets.ImageFolder(root=VAL_DATA, transform=data_transforms)
    test_dataset = datasets.ImageFolder(root=TEST_DATA, transform=data_transforms)
except FileNotFoundError:
    logger.error(f"Directory structure not found. Ensure you process the data in database/process_database.py")

print(f"Classes found: {train_dataset.classes}")
print(f"Total training images available: {len(train_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        
        # Load VGG19 with pre-trained weights
        self.model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Replace final layer with correct output classes
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, 7)

    def forward(self, x):
        return self.model(x)

class DriverDistractionModel(nn.Module):
    def __init__(self):
        super(DriverDistractionModel, self).__init__()
        self.flatten = nn.Flatten()


        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2), # 256x256 -> 128x128

            nn.Conv2d(16, 32, kernel_size=3, stride=1), 
            nn.ReLU(),
            nn.MaxPool2d(2), # 128x128 -> 64x64
        )

        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 62 * 62, 128), # 32 filters, 62x62 pixels, 128 neurons
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 2) # 128 neurons, 2 outputs
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classify(x)
        return x

class EarlyStopping:
    def __init__(self, patience = 20):
        self.patience = patience # how many batches without improvement to allow
        self.counter = 0 # num batches w/o improvement
        self.best_loss = float('inf') # best loss
        self.early_stop = False

    def __call__(self, loss):
        if loss < self.best_loss: # if loss improved (got smaller)
            self.best_loss = loss # update best loss
            self.counter = 0 # reset counter
            return self.early_stop, True
        else: # if loss didn't improve
            self.counter += 1 # increment counter
            if self.counter >= self.patience: # if counter exceeds patience
                self.early_stop = True # early stop
        return self.early_stop, False

def train_loop(dataloader, model, loss_fn, optimizer, epoch, best_loss, writer, device, early_stop):    
    print(f"\n--- Training Epoch {epoch+1} ---")

    model.train()
    start_time = time.time()

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("Loss/train", loss.item(), batch)
        
        # print(f"Batch {batch}: Loss = {loss.item():>7f}")

        should_stop, improved = early_stop(loss.item())

        if improved:
            best_loss = loss

            print("New best model found! Loss: ", loss.item(), " Saving...")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, MODEL_PATH)

        print(f"Batch {batch}: Loss = {loss.item():>7f}")

        if should_stop:
            return model, early_stop.best_loss, True
        
    end_time = time.time()
    print(f"Epoch {epoch+1} completed: {batch+1} batches processed")
    print(f"Time taken for epoch {epoch+1}: {end_time - start_time:.2f} seconds")
    return model, best_loss, False

def evaluate(dataloader, model, loss_fn, writer, device):
    print("\n--- Evaluating Model ---")

    test_loss, correct, total = 0, 0, 0

    model.eval()

    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total += len(y)
            test_loss += loss_fn(pred, y).item()
            correct += int((pred.argmax(1) == y).type(torch.float).sum().item())
            if batch == 9: break
    
    writer.add_scalar("Loss/test", test_loss / total)


    print("Total Samples: ", total)
    print("Correct Predictions: ", correct)
    print(f"Test Loss: {test_loss / total:.4f}")
    print(f"Evaluation: Accuracy = {int(100 * correct / total)}%" )

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on: ", device)

    print()
    print("--- Tensorboard Setup---")
    writer = SummaryWriter(LOG_DIR)

    print()
    print("--- Instantiate Model ---")
    #model = DoggoModel()
    model = VGG19().to(device)
    best_loss = float('inf')
    
    print("Adding graph to tensorboard...")
    dummy_data = torch.randn(1, 3, 256, 256).to(device)
    writer.add_graph(model, dummy_data)

    NUM_EPOCHS = 1
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=0.001
    )
    criterion = nn.CrossEntropyLoss()

    early_stop = EarlyStopping()

    print("--- Load Best Model ---")
    if os.path.exists(MODEL_PATH):
        best_model = torch.load(MODEL_PATH, weights_only=True)
        model.load_state_dict(best_model['model_state_dict'])
        optimizer.load_state_dict(best_model['optimizer_state_dict'])
        best_loss = best_model['loss']
        print("Loaded best model from ", MODEL_PATH)

    for epoch in range(NUM_EPOCHS):
        model, best_loss, early_stopped = train_loop(train_loader, model, criterion, optimizer, epoch, best_loss, writer, device, early_stop)
        evaluate(test_loader, model, criterion, writer, device)

        if early_stopped:
            print("Broke early")
            break

if __name__ == "__main__":
    main()