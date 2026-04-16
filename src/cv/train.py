import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import VGG19, EarlyStopping
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--patience", type=int, default=20)
parser.add_argument("--lr_patience",type=int, default=5)
parser.add_argument("--lr_factor", type=float, default=0.5)
args = parser.parse_args()


def main():
    training_dir = os.environ["SM_CHANNEL_TRAINING"]
    val_dir = os.environ["SM_CHANNEL_VALIDATION"]
    model_dir = os.environ["SM_MODEL_DIR"]

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(45),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(training_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)

    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    train_loader = DataLoader( train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True ) 
    val_loader = DataLoader( val_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    model = VGG19().to(device)
    loss_fn = nn.CrossEntropyLoss()

    # Only optimize the unfrozen classifier head
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    # Reduce LR when val loss plateaus
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        verbose=True
    )

    early_stop = EarlyStopping(patience=args.patience)
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(args.epochs):

        # ── Training ──────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # ── Validation ────────────────────────────────────────────
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += loss_fn(pred, y).item()
                total += len(y)
                correct += int((pred.argmax(1) == y).sum().item())

        val_loss /= len(val_loader)
        #val_loss /= total
        accuracy = 100 * correct / total
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"epoch {epoch:>3} | "
            f"train loss: {train_loss:.4f} | "
            f"val loss: {val_loss:.6f} | "
            f"val acc: {accuracy:.1f}% | "
            f"lr: {current_lr:.6f}"
        )

        # Step the LR scheduler
        scheduler.step(val_loss)

        # Save model if val loss improved
        should_stop, improved = early_stop(val_loss)
        if improved:
            print("New best model — saving checkpoint")
            torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))

        if should_stop:
            print("Early stopping triggered")
            break

    print("Training complete.")


if __name__ == "__main__":
    main()