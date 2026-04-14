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
