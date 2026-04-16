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
    
class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        
        # Load VGG19 with pre-trained weights
        self.model = models.vgg19(weights=models.VGG19_Weights.DEFAULT) 

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze block 5 (features indices 28–36) and full classifier
        for name, param in self.model.named_parameters():
            if any(f"features.{i}" in name for i in range(28, 37)):
                param.requires_grad = True
        
        # Replace entire classifier head with deeper head + dropout
        self.model.classifier = nn.Sequential(
            nn.Linear(25088, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        return self.model(x)

class EarlyStopping:
    def __init__(self, patience = 20):
        self.patience = patience # how many epochs without improvement to allow
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
